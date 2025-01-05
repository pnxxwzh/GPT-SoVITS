import xml.etree.ElementTree as ET
from typing import Dict

class ConfigManager:
    def __init__(self, config_path: str = "config/character_config.xml"):
        self.config_path = config_path
        self.tree = ET.parse(config_path)
        self.root = self.tree.getroot()
        self.user_name = self.root.find(".//user_name").text
        self.assistant_name = self.root.find(".//assistant_name").text
        self.assistant_role = self.root.find(".//assistant_role").text
        self.assistant_personality = self.root.find(".//assistant_personality").text
        self.relationship = self.root.find(".//relationship").text
        self.assistant_appearance = self.root.find(".//assistant_appearance").text
        self.environment = self.root.find(".//environment").text
        self.assistant_action = self.root.find(".//assistant_action").text