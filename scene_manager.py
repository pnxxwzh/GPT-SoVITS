import requests
from typing import List, Dict
import re
from config_manager import ConfigManager
import jieba

class SceneManager:
    def __init__(self, 
                 llm_url="http://localhost:1234/v1/chat/completions",
                 config_path: str = "config/character_config.xml"):
        self.llm_url = llm_url
        
        # 从配置文件加载称谓
        config = ConfigManager(config_path)
        user_name = config.user_name
        assistant_name = config.assistant_name
        environment = config.environment
        # 设置称谓映射
        self.perspective_mapping = {
            "我": user_name,
            "你": assistant_name,
            "我们": user_name + "和" + assistant_name,
            "我的": user_name + "的",
            "你的": assistant_name + "的",
        }
        
        # 初始化场景
        self.current_scene = environment
        self.scene_history = [self.current_scene]
        
        # 构建系统提示
        self.system_prompt = """你是一个专注于场景整合的AI助手。
## 你的任务是：
- 将新的场景描述与当前场景进行整合，输出最新的场景状态
- 保留重要的环境细节
- 解决可能的冲突
- 确保场景描述使用第三人称视角
- 输出简洁自然的描述，不添加输入不包含的细节，不超过200字
- 绝不添加输入不存在的细节与描述

## 示例1：
用户输入："当前场景：A与B在教室中，他们在认真学习，环境非常安静\n新增描述：老师打开了窗帘，阳光照进房间，A拿起一本书为B遮住阳光"
场景描述："A与B在充满阳光的教室中，A拿起一本书为B遮住阳光，环境非常安静"
## 示例2：
用户输入："当前场景：A与B在教室中，他们在认真学习，环境非常安静\n新增描述：下节课是体育课，A和B一起来到了操场上"
场景描述："A和B来到了操场上，他们正在上体育课"

请直接输出整合后的场景描述，不要包含任何其他内容。"""

    def set_perspective_mapping(self, mapping: Dict[str, str]) -> None:
        """更新称谓映射"""
        self.perspective_mapping.update(mapping)
    
    def convert_perspective(self, text: str) -> str:
        """使用jieba分词进行称谓替换"""
        # 将文本分词
        words = list(jieba.cut(text))
        
        # 替换匹配的词
        for i, word in enumerate(words):
            print(f"当前单词: {word}")
            if word in self.perspective_mapping:
                words[i] = self.perspective_mapping[word]
                print(f"替换成功: {word} -> {words[i]}")
        
        # 重新组合文本
        result = ''.join(words)
        
        if result == text:
            print("警告：没有进行任何替换")
        else:
            print(f"最终结果: {result}")
            
        return result
    
    def integrate_scene(self, new_scene: str) -> str:
        """使用辅助LLM整合环境信息"""
        # 先转换称谓
        converted_scene = self.convert_perspective(new_scene)

        print(f"转换后的场景描述: {converted_scene}")
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"当前场景：{self.current_scene}\n新增描述：{converted_scene}"}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                print(f"环境整合失败，使用新环境描述")
                return converted_scene
                
        except Exception as e:
            print(f"环境整合时出错: {e}")
            return converted_scene
    
    def update_scene(self, scene_description: str) -> str:
        """更新当前场景"""
        if scene_description:
            integrated_scene = self.integrate_scene(scene_description)
            self.current_scene = integrated_scene
            self.scene_history.append(integrated_scene)
            return integrated_scene
        return self.current_scene
    
    def get_current_scene(self) -> str:
        """获取当前场景"""
        return self.current_scene 