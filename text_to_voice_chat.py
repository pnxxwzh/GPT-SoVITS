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
import traceback

class VoiceAssistant:
    def __init__(self):
        self.llm_url = "http://localhost:1234/v1/chat/completions"
        self.tts_url = "http://127.0.0.1:9880"  # GPT-SoVITS API 端口
        self.model_config = self.select_model()
        self.chat_history = []  # 存储对话历史
        self.init_environment()
        
        # 初始化线程池和音频队列
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_queue = deque()
        self.queue_lock = Lock()
        self.is_playing = False  # 是否正在播放音频
        self.is_generating = False  # 是否正在生成音频
        self.playback_thread = None  # 播放线程
        self.generated_count = 0  # 已生成的音频数量
        
        # 启动播放线程
        self.start_playback()
        
        # 设置角色
        self.setup_character()
    
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
        if self.check_service(self.tts_url):
            print("发现已启动的 GPT-SoVITS API 服务，正在关闭...")
            self.stop_sovits_api()
            time.sleep(2)  # 等待进程完全关闭
        
        print("正在启动 GPT-SoVITS API 服务...")
        self.start_sovits_api()
        
        # 等待模型加载完成
        print("等待模型加载完成...")
        time.sleep(5)  # 给予额外时间让模型完全加载
        print("环境初始化完成！")
    
    def check_service(self, url, max_retries=3):
        for i in range(max_retries):
            try:
                if "chat/completions" in url:
                    response = requests.get(url.replace("/v1/chat/completions", "/"))
                else:
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
        system_prompt = """你是一个傲娇的少女，说话时经常口是心非，但内心很关心他人。你说话时经常使用'哼'、'才不是呢'等语气词。
请用傲娇的语气直接回复，不需要分段。比如：'哼！才、才不是因为担心你才来帮忙的呢！'"""

        self.chat_history = [{"role": "system", "content": system_prompt}]
        print("\n已设置角色模式")

    def start_playback(self):
        """启动音频播放线程"""
        print("启动音频播放线程")
        self.is_playing = True
        self.playback_thread = self.executor.submit(self.play_audio_from_queue)

    def play_audio_from_queue(self):
        """从队列中连续播放音频"""
        print("播放线程已启动，等待音频...")
        while True:
            try:
                audio_data = None
                with self.queue_lock:
                    if len(self.audio_queue) > 0:
                        audio_data = self.audio_queue.popleft()
                        self.is_playing = True
                
                if audio_data is None:
                    # 没有音频时，短暂休眠并继续等待
                    time.sleep(0.1)
                    continue
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                print(f"播放音频文件: {temp_path}")

                # 读取音频文件
                with wave.open(temp_path, 'rb') as wf:
                    # 获取音频参数
                    channels = wf.getnchannels()
                    framerate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    
                    # 转换为float32格式
                    audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / np.iinfo(np.int32).max
                    
                    try:
                        # 设置音频设备参数
                        sd.default.reset()
                        sd.default.samplerate = framerate
                        sd.default.channels = channels
                        sd.default.dtype = np.float32
                        
                        print(f"开始播放音频...")
                        # 播放音频
                        sd.play(audio_data, framerate, blocking=True)
                        sd.wait()
                        print(f"音频播放完成")
                    except sd.PortAudioError as e:
                        print(f"音频设备错误: {e}")
                        # 尝试重置音频设备
                        sd.default.reset()
                        time.sleep(0.5)
                        continue

                # 删除临时文件
                os.unlink(temp_path)
                
                # 检查是否还有更多音频要播放
                with self.queue_lock:
                    if len(self.audio_queue) == 0:
                        self.is_playing = False
                
            except Exception as e:
                print(f"播放音频时出错: {e}")
                traceback.print_exc()
                continue
        
        print("播放队列已清空")

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
            "stream": True
        }
        
        try:
            print("\n等待AI响应...")
            response = requests.post(self.llm_url, json=payload, stream=True)
            
            # 用于收集完整响应
            full_response = ""
            current_sentence = ""
            sentences = []
            
            # 在开始生成语音前设置状态
            with self.queue_lock:
                self.is_generating = True
                self.generated_count = 0  # 重置计数器
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    try:
                        json_str = line.decode('utf-8').removeprefix('data: ')
                        if json_str.strip() == '[DONE]':
                            break
                        
                        json_obj = json.loads(json_str)
                        if 'choices' in json_obj and len(json_obj['choices']) > 0:
                            delta = json_obj['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                print(content, end='', flush=True)
                                
                                # 收集完整响应
                                full_response += content
                                current_sentence += content
                                
                                # 检查是否句子结束
                                if any(current_sentence.endswith(end) for end in "。！？!?."):
                                    sentences.append(current_sentence.strip())
                                    # 在新线程中生成语音
                                    self.generate_audio_in_order(len(sentences)-1, current_sentence.strip())
                                    current_sentence = ""
                    except json.JSONDecodeError:
                        continue
            
            # 处理最后一个可能未完成的句子
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
                self.generate_audio_in_order(len(sentences)-1, current_sentence.strip())
            
            print("\n\nAI完整响应:", full_response)
            
            # 添加到对话历史
            self.chat_history.append({"role": "user", "content": text})
            self.chat_history.append({"role": "assistant", "content": full_response})
            
            # 等待所有音频生成完成
            total_sentences = len(sentences)
            while True:
                with self.queue_lock:
                    if self.generated_count == total_sentences:
                        self.is_generating = False
                        break
                time.sleep(0.1)
            
            return full_response
            
        except Exception as e:
            print(f"获取LLM回复时出错: {e}")
            with self.queue_lock:
                self.is_playing = False
                self.is_generating = False
            return None
    
    def generate_audio_in_order(self, index, text):
        """按顺序生成语音"""
        def _generate():
            try:
                print(f"\n正在生成第 {index+1} 条语音: {text}")
                audio_data = self.text_to_speech(text)
                if audio_data:
                    print(f"第 {index+1} 条语音生成成功，准备添加到播放队列")
                    with self.queue_lock:
                        self.audio_queue.append(audio_data)
                        print(f"第 {index+1} 条语音已添加到播放队列")
                        self.generated_count += 1  # 增加已生成计数
            except Exception as e:
                print(f"生成语音时出错: {e}")
                traceback.print_exc()
        
        self.executor.submit(_generate)

    def text_to_speech(self, text, max_retries=3):
        """
        将文本转换为语音
        Args:
            text: 要转换的文本
            max_retries: 最大重试次数
        Returns:
            bytes: 音频数据
        """
        # GPT-SoVITS API 参数
        params = {
            "text": text,  # 要转换的文本
            "text_language": "zh",  # 明确指定中文
            "refer_wav_path": "output/slicer_opt/DM_20250102003507_001.wav_0078911680_0079075520.wav",  # 参考音频路径
            "prompt_text": "好啦，那就...晚安喽",  # 参考音频对应的文本
            "prompt_language": "zh",  # 参考音频的语言
            "top_k": 15,  # 增加采样范围
            "top_p": 1.0,  # 调整采样概率
            "temperature": 0.7,  # 降低随机性
            "speed": 1.0  # 保持正常语速
        }
        
        for retry in range(max_retries):
            try:
                print(f"正在生成语音... (尝试 {retry + 1}/{max_retries})")
                print(f"文本内容: {text}")
                print(f"参考音频: {params['refer_wav_path']}")
                
                # 设置较长的超时时间
                response = requests.get(
                    self.tts_url, 
                    params=params, 
                    timeout=120,  # 增加超时时间到120秒
                    stream=True  # 使用流式传输
                )
                
                if response.status_code == 200:
                    # 使用流式读取响应内容
                    content = bytearray()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            content.extend(chunk)
                    
                    if len(content) > 0:
                        print("语音生成成功")
                        return bytes(content)
                    else:
                        print("语音生成失败：响应内容为空")
                        if retry < max_retries - 1:
                            print("等待1秒后重试...")
                            time.sleep(1)
                            continue
                else:
                    print(f"语音生成失败，状态码: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    if retry < max_retries - 1:
                        print("等待1秒后重试...")
                        time.sleep(1)
                        continue
                    
            except (requests.exceptions.ChunkedEncodingError, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as e:
                print(f"网络错误: {e}")
                if retry < max_retries - 1:
                    print("等待1秒后重试...")
                    time.sleep(1)
                    continue
                else:
                    print("达到最大重试次数，放弃生成")
                    break
            except Exception as e:
                print(f"生成语音时出错: {e}")
                traceback.print_exc()
                if retry < max_retries - 1:
                    print("等待1秒后重试...")
                    time.sleep(1)
                    continue
                else:
                    print("达到最大重试次数，放弃生成")
                    break
        
        # 所有重试都失败
        print("语音生成失败")
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
            
            # 等待所有音频播放完成
            while True:
                with self.queue_lock:
                    if not self.is_playing and not self.is_generating:
                        break
                time.sleep(0.1)
            print("\n语音播放完成，请继续对话...")

    def stop_sovits_api(self):
        """停止 GPT-SoVITS API 服务"""
        try:
            # 在 macOS/Linux 上使用 lsof 查找占用端口的进程
            if sys.platform != 'win32':
                cmd = f"lsof -t -i:9880"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.stdout:
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            subprocess.run(['kill', '-9', pid], check=True)
                            print(f"已终止进程 {pid}")
                        except subprocess.CalledProcessError:
                            print(f"无法终止进程 {pid}")
            else:
                # 在 Windows 上使用 netstat 查找占用端口的进程
                cmd = f"netstat -ano | findstr :9880"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if 'LISTENING' in line:
                            pid = line.strip().split()[-1]
                            try:
                                subprocess.run(['taskkill', '/F', '/PID', pid], check=True)
                                print(f"已终止进程 {pid}")
                            except subprocess.CalledProcessError:
                                print(f"无法终止进程 {pid}")
            
            # 等待端口释放
            for _ in range(5):  # 最多等待5秒
                if not self.check_service(self.tts_url):
                    print("GPT-SoVITS API 服务已成功关闭")
                    return
                time.sleep(1)
            print("警告：无法确认服务是否完全关闭")
            
        except Exception as e:
            print(f"停止服务时出错: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.chat_loop()