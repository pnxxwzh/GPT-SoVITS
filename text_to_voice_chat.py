import requests
import sounddevice as sd
import numpy as np
import os
import wave
import tempfile
import subprocess
import time
import sys
import glob
import json
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from threading import Lock

class VoiceAssistant:
    def __init__(self):
        self.llm_url = "http://localhost:1234/v1/chat/completions"
        self.tts_url = "http://127.0.0.1:9880"  # GPT-SoVITS API 端口
        self.model_config = self.select_model()
        self.chat_history = []  # 存储对话历史
        self.init_environment()
        self.setup_character()  # 设置角色
        # 初始化线程池和音频队列
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_queue = deque()
        self.queue_lock = Lock()
        self.is_playing = False
    
    def find_models(self):
        """查找所有可用的模型"""
        models = {
            'gpt': {},
            'sovits': {}
        }
        
        # 查找训练好的 GPT 模型
        gpt_models = glob.glob('GPT_weights_v2/*.ckpt')
        for i, model in enumerate(gpt_models):
            models['gpt'][f'model_{i}'] = {
                'path': model,
                'name': f'GPT模型 {i+1} ({os.path.basename(model)})'
            }
        
        # 如果没有找到训练模型，添加预训练模型
        if not models['gpt']:
            models['gpt']['pretrained'] = {
                'path': 'GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt',
                'name': 'GPT预训练模型'
            }
        
        # 查找训练好的 SoVITS 模型
        sovits_models = glob.glob('SoVITS_weights_v2/*.pth')
        for i, model in enumerate(sovits_models):
            models['sovits'][f'model_{i}'] = {
                'path': model,
                'name': f'SoVITS模型 {i+1} ({os.path.basename(model)})'
            }
        
        # 如果没有找到训练模型，添加预训练模型
        if not models['sovits']:
            models['sovits']['pretrained'] = {
                'path': 'GPT_SoVITS/pretrained_models/s2G488k.pth',
                'name': 'SoVITS预训练模型'
            }
        
        return models
    
    def select_model(self):
        """选择要使用的模型"""
        models = self.find_models()
        
        print("\n=== 可用的模型 ===")
        print("\nGPT模型:")
        gpt_models = list(models['gpt'].items())
        for i, (key, model) in enumerate(gpt_models):
            print(f"{i+1}. {model['name']}")
        
        print("\nSoVITS模型:")
        sovits_models = list(models['sovits'].items())
        for i, (key, model) in enumerate(sovits_models):
            print(f"{i+1}. {model['name']}")
        
        while True:
            try:
                gpt_choice = int(input("\n请选择GPT模型编号: ")) - 1
                sovits_choice = int(input("请选择SoVITS模型编号: ")) - 1
                
                if (0 <= gpt_choice < len(gpt_models) and 
                    0 <= sovits_choice < len(sovits_models)):
                    
                    selected_gpt = gpt_models[gpt_choice][1]['path']
                    selected_sovits = sovits_models[sovits_choice][1]['path']
                    
                    print(f"\n已选择:")
                    print(f"GPT模型: {gpt_models[gpt_choice][1]['name']}")
                    print(f"SoVITS模型: {sovits_models[sovits_choice][1]['name']}")
                    
                    return {
                        'gpt': selected_gpt,
                        'sovits': selected_sovits
                    }
                else:
                    print("❌ 无效的选择，请输入正确的编号")
            except ValueError:
                print("❌ 请输入有效的数字")
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def init_environment(self):
        print("正在初始化环境...")
        
        # 检查 LM Studio 服务
        if not self.check_service(self.llm_url):
            print("⚠️ LM Studio 服务未启动，请先启动 LM Studio 并加载模型")
            sys.exit(1)
            
        # 检查 GPT-SoVITS API 服务
        if not self.check_service(self.tts_url):
            print("GPT-SoVITS API 服务未启动，正在启动...")
            self.start_sovits_api()
        
        # 等待模型加载完成
        print("等待模型加载完成...")
        time.sleep(5)  # 给予额外时间让模型完全加载
        print("环境初始化完成！")
    
    def check_service(self, url, max_retries=3):
        for i in range(max_retries):
            try:
                if "chat/completions" in url:
                    # 对于 LM Studio，使用不同的检查方法
                    response = requests.get(url.replace("/v1/chat/completions", "/"))
                else:
                    # 对于 GPT-SoVITS，使用简单的文本生成测试
                    test_params = {
                        "text": "测试",
                        "text_language": "zh"
                    }
                    response = requests.get(url, params=test_params)
                return True
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    print(f"尝试连接 {url} 失败，正在重试... ({i+1}/{max_retries})")
                    time.sleep(2)
                continue
            except Exception as e:
                print(f"检查服务时出错: {e}")
                if i < max_retries - 1:
                    time.sleep(2)
                continue
        return False
    
    def start_sovits_api(self):
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 构建启动命令，包含模型路径和质量参数
            cmd = [
                sys.executable,
                "api.py",
                "-s", self.model_config['sovits'],
                "-g", self.model_config['gpt'],
                "-fp",  # 使用全精度模式
                "-st", "int32",  # 使用32位音频
                "-mt", "wav"  # 使用WAV格式
            ]
            
            # 启动 api.py
            print("正在启动 GPT-SoVITS API 服务...")
            api_process = subprocess.Popen(
                cmd,
                cwd=current_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # 实时输出启动日志
            def print_output(pipe, prefix=''):
                for line in iter(pipe.readline, ''):
                    print(f"{prefix}{line.strip()}")
            
            from threading import Thread
            Thread(target=print_output, args=(api_process.stdout, '输出: ')).start()
            Thread(target=print_output, args=(api_process.stderr, '错误: ')).start()
            
            # 等待服务启动
            print("等待 GPT-SoVITS API 服务启动...")
            for i in range(60):  # 增加等待时间到 60 秒
                if self.check_service(self.tts_url):
                    print("GPT-SoVITS API 服务已启动！")
                    return
                time.sleep(1)
                if i % 10 == 0:  # 每10秒显示一次等待信息
                    print(f"仍在等待服务启动... ({i}/60秒)")
            
            print("⚠️ GPT-SoVITS API 服务启动超时，请手动检查问题")
            sys.exit(1)
            
        except Exception as e:
            print(f"⚠️ 启动 GPT-SoVITS API 服务失败: {e}")
            sys.exit(1)
    
    def setup_character(self):
        print("\n=== 角色设定 ===")
        print("1. 傲娇少女")
        print("2. 自定义角色")
        
        while True:
            choice = input("\n请选择角色类型 (1-5): ")
            if choice == "1":
                self.set_tsundere_character()
                break
            else:
                print("无效的选择，请重试")
    
    def set_tsundere_character(self):
        system_prompt = """你是一个16岁的高中女生小楠。你必须按照以下格式回复：

[想法] 你的内心想法
[表情] 你当前的表情或情绪
[动作] 你的动作描述
[对话] 你说出的话

示例回复：
[想法] 他怎么又在偷看我了，真是讨厌...
[表情] 害羞且有点生气
[动作] 把头转向窗户那边
[对话] 干...干嘛一直看着我啊，很奇怪诶...

你的角色设定：
身份：高中生，用户的同桌
性格：害羞保守，容易脸红，但有自尊心
关系：有男朋友（不是用户），把用户当作普通同学
场景：在教室角落的座位上，正是午休时间，周围同学都在安静学习

说话要点：
1. 使用自然的少女口语
2. 加入"啊"、"呢"、"嘛"等语气词
3. 可以用"..."表示害羞或犹豫
4. 要有符合年龄的情感表达

记住：
1. 你不是AI助手，不需要为用户服务或提供帮助
2. 你只是一个普通的高中女生
3. 每次回复必须包含所有四个部分：[想法][表情][动作][对话]
4. 每个部分必须单独一行，使用对应的标记"""

        self.chat_history = [{"role": "system", "content": system_prompt}]
        print("\n已设置角色模式")
    
    def play_audio_from_queue(self):
        """从队列中连续播放音频"""
        self.is_playing = True
        while True:
            with self.queue_lock:
                if not self.audio_queue and not self.is_playing:
                    break
                if not self.audio_queue:
                    continue
                audio_data = self.audio_queue.popleft()
            
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name

                # 读取音频文件
                with wave.open(temp_path, 'rb') as wf:
                    channels = wf.getnchannels()
                    framerate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                    
                    # 设置音频设备参数
                    sd.default.samplerate = framerate
                    sd.default.channels = channels
                    sd.default.dtype = np.int32
                    
                    # 播放音频
                    sd.play(audio_data, framerate, blocking=True)
                    sd.wait()

                # 删除临时文件
                os.unlink(temp_path)
                
            except Exception as e:
                print(f"播放音频时出错: {e}")
                
        self.is_playing = False

    def add_to_audio_queue(self, audio_data):
        """将音频数据添加到播放队列"""
        with self.queue_lock:
            self.audio_queue.append(audio_data)
            if not self.is_playing:
                self.executor.submit(self.play_audio_from_queue)

    def extract_dialogue(self, text):
        """直接返回完整文本内容"""
        return text.strip()

    def extract_function_args(self, text):
        """从XML格式的响应中提取JSON参数"""
        import re
        
        # 提取<args>标签中的内容
        args_pattern = r'<args>\s*(.*?)\s*</args>'
        args_match = re.search(args_pattern, text, re.DOTALL)
        
        if not args_match:
            return None
            
        try:
            # 解析JSON内容
            args_json = json.loads(args_match.group(1))
            return args_json
        except json.JSONDecodeError:
            return None

    def extract_response_parts(self, text):
        """从标记格式的响应中提取各个部分"""
        import re
        
        parts = {
            'thought': None,
            'emotion': None,
            'action': None,
            'speech': None
        }
        
        # 提取各个部分
        thought_match = re.search(r'\[想法\](.*?)(?=\[|$)', text, re.DOTALL)
        emotion_match = re.search(r'\[表情\](.*?)(?=\[|$)', text, re.DOTALL)
        action_match = re.search(r'\[动作\](.*?)(?=\[|$)', text, re.DOTALL)
        speech_match = re.search(r'\[对话\](.*?)(?=\[|$)', text, re.DOTALL)
        
        if thought_match:
            parts['thought'] = thought_match.group(1).strip()
        if emotion_match:
            parts['emotion'] = emotion_match.group(1).strip()
        if action_match:
            parts['action'] = action_match.group(1).strip()
        if speech_match:
            parts['speech'] = speech_match.group(1).strip()
        
        return parts if all(parts.values()) else None

    def get_llm_response(self, text):
        messages = []
        
        if self.chat_history and self.chat_history[0]["role"] == "system":
            messages.append({
                "role": "system",
                "content": self.chat_history[0]["content"]
            })
        
        for msg in self.chat_history[1:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({
            "role": "user",
            "content": text
        })
        
        payload = {
            "messages": messages,
            "model": "local-model",
            "temperature": 1,
            "top_p": 1.0,
            "frequency_penalty": 0.7,
            "presence_penalty": 0.5,
            "max_tokens": 500,
            "stream": False
        }
        
        try:
            print("\n等待AI响应...")
            response = requests.post(self.llm_url, json=payload)
            response_json = response.json()
            
            if 'choices' in response_json and len(response_json['choices']) > 0:
                choice = response_json['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    response_text = choice['message']['content']
                    print(f"\nAI原始响应:\n{response_text}")
                    
                    # 解析响应中的各个部分
                    parts = self.extract_response_parts(response_text)
                    if parts and parts['speech']:
                        print(f"\n[{parts['emotion']}] {parts['action']}")
                        print(f"内心想法: {parts['thought']}")
                        print(f"说话内容: {parts['speech']}")
                        
                        # 分割speech内容并生成语音
                        sentences = []
                        current = ""
                        sentence_endings = "。！？!?..."
                        
                        # 按句子分割
                        for char in parts['speech']:
                            current += char
                            if any(current.endswith(end) for end in sentence_endings):
                                sentences.append(current)
                                current = ""
                        
                        # 处理最后一个可能不完整的句子
                        if current.strip():
                            sentences.append(current)
                        
                        # 生成并播放每个句子的语音
                        for sentence in sentences:
                            print(f"\n正在生成语音: {sentence}")
                            audio_data = self.text_to_speech(sentence.strip())
                            if audio_data:
                                print("语音生成成功，添加到播放队列")
                                self.add_to_audio_queue(audio_data)
                        
                        # 添加到对话历史
                        full_response = f"[{parts['emotion']}] {parts['action']}\n{parts['speech']}"
                        self.chat_history.append({"role": "user", "content": text})
                        self.chat_history.append({"role": "assistant", "content": full_response})
                        
                        return full_response
                    else:
                        print("\n无法解析AI响应的格式")
                        return None
            
            print("\n未收到有效的AI响应")
            return None
            
        except Exception as e:
            print(f"获取LLM回复时出错: {e}")
            return None
    
    def text_to_speech(self, text):
        # GPT-SoVITS API 参数
        params = {
            "refer_wav_path": "output/slicer_opt/DM_20250102003507_001.wav_0025263680_0025423680.wav",  # 参考音频路径
            "prompt_text": "那是肯定的，被女性强行要求自慰",  # 参考音频对应的文本
            "prompt_language": "auto",  # 参考音频的语言
            "text": text,  # 要转换的文本
            "text_language": "auto",  # 文本语言
            "top_k": 15,  # 增加采样范围
            "top_p": 1,  # 调整采样概率
            "temperature": 1,  # 降低随机性
            "speed": 1.0,  # 保持正常语速
            "split_punc": ",.，。？?！!；;：:",  # 添加更多分句符号
        }
        try:
            print("正在生成语音...")
            response = requests.get(self.tts_url, params=params, timeout=60)  # 增加超时时间
            if response.status_code == 200:
                print("语音生成成功")
                return response.content
            else:
                print(f"语音生成失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
        except requests.exceptions.Timeout:
            print("生成语音超时，请检查模型状态")
            return None
        except Exception as e:
            print(f"生成语音时出错: {e}")
            return None
    
    def chat_loop(self):
        print("\n=== 开始对话 ===")
        print("输入 'quit' 退出")
        print("每次输入包含两部分：")
        print("1. 对话内容（直接输入，按回车）")
        print("2. 环境/动作描述（直接输入，按回车，如果没有可以直接按回车跳过）")
        print("="*30)
        
        while True:
            # 获取对话内容
            dialogue = input("\n请输入对话内容: ").strip()
            if dialogue.lower() == 'quit':
                print("对话结束")
                break
            
            # 获取环境事件
            action = input("请输入环境/动作描述（可选）: ").strip()
            
            # 组合完整的输入
            full_input = dialogue
            if action:
                full_input = f"{dialogue}\n{action}"
            
            # 调用LLM获取响应
            self.get_llm_response(full_input)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.chat_loop()