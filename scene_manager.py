import requests
from typing import List, Dict
import re
from config_manager import ConfigManager
import jieba
import time

class SceneManager:
    def __init__(self, 
                 llm_url="http://localhost:1234/v1/chat/completions",
                 config_path: str = "config/character_config.xml"):
        self.llm_url = llm_url
        self.base_url = llm_url.rsplit('/chat/completions', 1)[0]  # 提取基础URL
        
        # 获取可用模型
        self.available_models = self.get_available_models()
        if not self.available_models:
            raise ValueError("没有可用的模型")
        
        # 分别选择主对话模型和场景分析模型
        print("\n请选择主对话模型：")
        self.chat_model = self.select_model() if self.available_models else None
        print("\n请选择场景分析模型：")
        self.scene_model = self.select_model() if self.available_models else None
        
        # 从配置文件加载配置
        config = ConfigManager(config_path)
        self.user_name = config.user_name
        self.assistant_name = config.assistant_name
        self.environment = config.environment
        
        # 初始化场景
        self.current_scene = self.environment
        self.scene_history = [self.current_scene]
        
        # 大幅简化提示词，使其更直接
        self.scene_analysis_prompt = f"""
你是场景分析助手，任务是分析场景变化，并更新场景描述：
每次任务，你会收到当前场景、对话者，以及对话内容及发生的事情，请基于这些信息，更新场景描述。
更新过程请注意：
1. 场景描述要简短概括，但不要丢失信息，使用第三人称，不超过200字
2. 场景描述要准确，不要出现不存在的场景
3. 不要增加输入中不存在的信息

用一段话描述更新后的场景，不要输出任何其他内容，也不要推测故事的发展

"""

    def get_available_models(self) -> list:
        """获取当前可用的模型列表"""
        try:
            # 获取模型列表
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                models = response.json()
                available_models = []
                for model in models["data"]:
                    # 只获取模型ID
                    if model.get("id"):
                        available_models.append(model["id"])
                if available_models:
                    return available_models
                else:
                    print("警告：未找到可用的模型")
                    return []
            else:
                print(f"获取模型列表失败: HTTP {response.status_code}")
                return []
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return []

    def select_model(self) -> str:
        """让用户选择模型"""
        if not self.available_models:
            print("错误：没有可用的模型")
            return None
            
        print("\n可用模型列表:")
        for i, model in enumerate(self.available_models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = input("\n请选择要使用的模型 (输入序号): ")
                if choice.lower() == 'q':
                    print("退出模型选择")
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_models):
                    selected_model = self.available_models[idx]
                    print(f"已选择模型: {selected_model}")
                    return selected_model
                else:
                    print("无效的选择，请重试或输入 'q' 退出")
            except ValueError:
                print("请输入有效的数字，或输入 'q' 退出")

    def analyze_dialogue(self, dialogue: str, speaker: str, max_retries=3) -> str:
        """分析对话内容中的场景变化"""
        for retry in range(max_retries):
            try:
                source = self.user_name if speaker == "user" else self.assistant_name
                analysis_prompt = f'当前场景：{self.current_scene}\n对话者：{source}\n对话者说："{dialogue}"'

                print(f"正在进行第 {retry + 1} 次尝试...")
                response = requests.post(
                    self.llm_url,
                    json={
                        "model": self.scene_model,  # 使用场景分析模型
                        "messages": [
                            {"role": "system", "content": self.scene_analysis_prompt},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500,
                        "top_p": 0.9,
                        "presence_penalty": 0.6,
                        "frequency_penalty": 0.3
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()['choices'][0]['message']['content'].strip()
                    if result and result != "NO_CHANGE" and result != self.current_scene:
                        return result
                    break  # 如果响应成功但没有场景变化，就不需要重试
                
                print(f"第 {retry + 1} 次尝试失败，状态码: {response.status_code}")
                
            except requests.exceptions.Timeout:
                print(f"第 {retry + 1} 次尝试超时")
                if retry < max_retries - 1:
                    print("等待 2 秒后重试...")
                    time.sleep(2)
                continue
                
            except requests.exceptions.RequestException as e:
                print(f"第 {retry + 1} 次尝试出错: {e}")
                if retry < max_retries - 1:
                    print("等待 2 秒后重试...")
                    time.sleep(2)
                continue
                
            except Exception as e:
                print(f"第 {retry + 1} 次尝试发生未知错误: {e}")
                if retry < max_retries - 1:
                    print("等待 2 秒后重试...")
                    time.sleep(2)
                continue
        
        print(f"经过 {max_retries} 次尝试后仍然失败")
        return None

    def update_from_dialogue(self, dialogue: str, speaker: str) -> str:
        """根据对话内容更新场景"""
        # 打印调试信息
        print(f"\n正在分析对话: {dialogue}")
        print(f"说话者: {speaker}")
        
        new_scene = self.analyze_dialogue(dialogue, speaker)
        if new_scene:
            print(f"场景已更新: {new_scene}")
            self.current_scene = new_scene
            self.scene_history.append(new_scene)
            return new_scene
        
        print("场景未发生变化")
        return self.current_scene

    def update_scene(self, scene_description: str) -> str:
        """手动更新场景（保留原有功能）"""
        if scene_description:
            self.current_scene = scene_description
            self.scene_history.append(scene_description)
            return scene_description
        return self.current_scene
    
    def get_current_scene(self) -> str:
        """获取当前场景"""
        return self.current_scene 

    def get_chat_model(self) -> str:
        """获取当前主对话模型"""
        return self.chat_model

    def get_scene_model(self) -> str:
        """获取当前场景分析模型"""
        return self.scene_model 