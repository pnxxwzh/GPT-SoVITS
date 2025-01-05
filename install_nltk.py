import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 下载必要的 NLTK 数据
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('cmudict')
nltk.download('tagsets') 