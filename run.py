from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from PIL import Image
from time import sleep
import pandas as pd
import numpy as np
import keyboard 
import os
import requests
import json
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from mcan_bert import NetShareFusion, process_dct_img  # 从训练代码中导入模型定义和处理函数

class InferenceDataset(Dataset):
    def __init__(self, texts, images, VOCAB_DIR, max_sen_len, transform_vgg=None, transform_dct=None):
        super(InferenceDataset, self).__init__()
        self.transform_vgg = transform_vgg
        self.transform_dct = transform_dct
        self.tokenizer = BertTokenizer.from_pretrained(VOCAB_DIR)
        self.max_sen_len = max_sen_len
        self.texts = texts
        self.images = images

    def __getitem__(self, idx):
        content = str(self.texts[idx])
        text_content = self.tokenizer.encode_plus(content, add_special_tokens=True, padding='max_length',
                                                  truncation=True, max_length=self.max_sen_len, return_tensors='pt')

        # 确保图像转换后的通道数为3
        image = self.transform_vgg(self.images[idx].convert('RGB'))
        dct_img = self.transform_dct(self.images[idx].convert('L'))
        dct_img = process_dct_img(dct_img)

        return {
            "text_input_ids": text_content["input_ids"].flatten().clone().detach().type(torch.LongTensor),
            "attention_mask": text_content["attention_mask"].flatten().clone().detach().type(torch.LongTensor),
            "token_type_ids": text_content["token_type_ids"].flatten().clone().detach().type(torch.LongTensor),
            "image": image,
            "dct_img": dct_img,
        }

    def __len__(self):
        return len(self.texts)
    

class InferenceModel():
    def __init__(self):
        self.model_path = 'result/model_save/model.pth'
        self.vocab_dir = 'models/bert-base-chinese/'
        self.pthfile_path = 'models/vgg19-dcbb9e9d.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self.config = {
            "CASED": self.vocab_dir,
            "pthfile": self.pthfile_path,
            "kernel_sizes": [3, 3, 3],
            "num_channels": [32, 64, 128],
            "num_layers": 4,
            "num_heads": 4,
            "model_dim": 256,
            "dropout": 0.5,
            "drop_and_BN": 'drop-BN',
            "max_sen_len": 256
        }
        self.model = self.load_model(self.model_path, self.config)
        
    def load_model(self, model_path, config):
        # 加载预训练的BERT模型架构
        bert_model = BertModel.from_pretrained(config['CASED'])
        # 初始化NetShareFusion模型
        model = NetShareFusion(
            CASED=config['CASED'],
            pthfile=config['pthfile'],
            kernel_sizes=config['kernel_sizes'],
            num_channels=config['num_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            model_dim=config['model_dim'],
            dropout=config['dropout'],
            drop_and_BN=config['drop_and_BN']
        )
        # 加载训练好的模型权重
        model.load_state_dict(torch.load(model_path, map_location='cpu')['net'],strict=False)
        model.eval()
        return model
    
    def predict(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    'text_input_ids': batch['text_input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'token_type_ids': batch['token_type_ids'].to(self.device),
                    'image': batch['image'].to(self.device),
                    'dct_img': batch['dct_img'].to(self.device),
                    'attn_mask': None
                }

                outputs = self.model(**inputs)
                probs = torch.softmax(outputs[1], dim=1).cpu().numpy()
                predictions.extend(probs)

        return predictions
    

class Crawler():
    def __init__(self):
        requests.adapters.DEFAULT_RETRIES = 5
        requests.packages.urllib3.disable_warnings()
        
        self.output_folder = 'test_images'
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.options = Options()
        self.options.add_argument('-ignore-certificate-errors')
        self.options.add_argument('-ignore -ssl-errors')
        self.options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
        self.service = Service(executable_path='edgedriver_win64\msedgedriver.exe')
        self.url = "https://m.weibo.cn"
        self.data = None
        
        print("init successfully")
        
    def download_image(self, url, save_path):
        if (url == -1):
            return
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            # print(f"Downloaded successfully: {url} -> {save_path}")
        # else:
            # print(f"Download failed: {url}")
            
    def fetch_and_crawl(self):
        # 获取当前URL
        url = self.driver.current_url
        # print("当前 URL:", url)
        # 等待页面完全加载
        self.driver.implicitly_wait(10)
        # 获取页面的HTML内容
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        # 找到微博正文部分的元素
        content = soup.find('div', class_='weibo-text')  # 类名根据实际页面修改
        # 假设评论总是位于<h3>标签中
        comments = soup.find_all('h3')
        comment_text = []
        for comment in comments:
            comment_text.append(comment.get_text(strip=True))
        # 获取页面源代码
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        pic_url = soup.find_all("img", class_='f-bg-img')
        if len(pic_url) == 0:
            pic_url = -1
        else:
            pic_url = pic_url[0]['src']
        self.data = {"content": content.get_text(strip=True), "commments": comment_text, "pic_url": pic_url}

    def data_process(self, data):
        save_path = os.path.join(self.output_folder, "tmp.jpg")
        self.download_image(data['pic_url'], save_path)
        if data['pic_url'] == -1:
            input_image = np.full((224, 224, 3), 127, dtype=np.uint8)
            input_image = Image.fromarray(input_image)
        else:
            input_image = Image.open(save_path)
        transform_vgg = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        transform_dct = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        return data['content'], input_image, transform_vgg, transform_dct
        # 待补充 emotion
        # dataset = InferenceDataset(texts=[input_text], images=[input_image], VOCAB_DIR=vocab_dir, max_sen_len=config['max_sen_len'],
        #                        transform_vgg=transform_vgg, transform_dct=transform_dct)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        
def exit_prog():
    crawler.driver.quit()
    global running
    running = False
    print("退出")

crawler = Crawler()
model = InferenceModel()
keyboard.add_hotkey('p', crawler.fetch_and_crawl)
keyboard.add_hotkey('esc', exit_prog)
running = True

def run(crawler, model):
    crawler.driver = webdriver.Edge(service=crawler.service, options=crawler.options)
    crawler.driver.maximize_window()
    crawler.driver.get(crawler.url)
    # running = True
    
    js_code = """
    var button = document.createElement('button');
    button.innerHTML = 'Click Me';

    // 自定义按钮样式
    button.innerHTML = '检验真假';
    button.style.position = 'fixed';
    button.style.top = '400px';
    button.style.right = '50px';
    button.style.border = '1px solid #ccc';
    button.style.borderRadius = '5px';
    button.style.cursor = 'pointer'
    button.style.fontSize = '25px'
    button.style.backgroundColor = '#f0f0f0'

    // 添加点击事件
    button.addEventListener('click', function() {
        // 如果 resultDiv 已经存在，先删除它
        var existingDiv = document.getElementById('resultDiv');
        if (existingDiv) {
            existingDiv.parentNode.removeChild(existingDiv);
        }

        var triggerElement = document.createElement('div');
        triggerElement.id = 'pythonTrigger';
        document.body.appendChild(triggerElement);
    });

    document.body.appendChild(button);
    """
    
    # 等待触发元素 '#pythonTrigger' 出现
    while running:
        # try:
            crawler.driver.execute_script(js_code)
            WebDriverWait(crawler.driver, 10000).until(
                EC.presence_of_element_located((By.ID, "pythonTrigger"))
            )
            # if crawler.data != None:
            #     input_text, input_image, transform_vgg, transform_dct = crawler.data_process(crawler.data)
            #     crawler.data = None
            #     dataset = InferenceDataset(texts=[input_text], images=[input_image], VOCAB_DIR=model.vocab_dir, max_sen_len=model.config['max_sen_len'], transform_vgg=transform_vgg, transform_dct=transform_dct)
            #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            #     predictions = model.predict(dataloader)
            #     print(f"真概率: {predictions[0][1]}, 假概率: {predictions[0][0]}")
            sleep(1)
            
            # 调用Python函数并获取结果
            js_code_display_first_popup = """
            var firstPopup = document.createElement('div');
            firstPopup.id = 'firstPopup';
            firstPopup.innerHTML = '判定中，请稍等';
            firstPopup.style.position = 'fixed';
            firstPopup.style.top = '450px';
            firstPopup.style.right = '50px';
            firstPopup.style.padding = '10px';
            firstPopup.style.backgroundColor = '##f0f0f0';
            firstPopup.style.border = '1px solid #ccc';
            firstPopup.style.borderRadius = '5px';
            firstPopup.style.fontSize = '14px';
            document.body.appendChild(firstPopup);
            """
            crawler.driver.execute_script(js_code_display_first_popup)
            crawler.fetch_and_crawl()           
            input_text, input_image, transform_vgg, transform_dct = crawler.data_process(crawler.data)
            crawler.data = None
            dataset = InferenceDataset(texts=[input_text], images=[input_image], VOCAB_DIR=model.vocab_dir, max_sen_len=model.config['max_sen_len'], transform_vgg=transform_vgg, transform_dct=transform_dct)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            predictions = model.predict(dataloader)
            # sleep(5)
            # 使用JavaScript将结果插入到页面中，并添加关闭按钮
            js_code_display_second_popup = f"""
            // 关闭第一个弹窗
            var firstPopup = document.getElementById('firstPopup');
            if (firstPopup) {{
                firstPopup.parentNode.removeChild(firstPopup);
            }}

            // 显示第二个弹窗
            var resultDiv = document.createElement('div');
            resultDiv.id = 'resultDiv';
            resultDiv.innerHTML = '{f"真概率: {predictions[0][1]}, 假概率: {predictions[0][0]}"} <button id="closeBtn">Close</button>';
            resultDiv.style.position = 'fixed';
            resultDiv.style.top = '450px';
            resultDiv.style.right = '50px';
            resultDiv.style.padding = '10px';
            resultDiv.style.backgroundColor = '#f0f0f0';
            resultDiv.style.border = '1px solid #ccc';
            resultDiv.style.borderRadius = '5px';
            resultDiv.style.fontSize = '14px';
            document.body.appendChild(resultDiv);

            // 添加关闭按钮的点击事件
            var closeButton = document.getElementById('closeBtn');
            closeButton.style.marginLeft = '10px';
            closeButton.style.padding = '5px';
            closeButton.style.backgroundColor = '#d9534f';
            closeButton.style.color = 'white';
            closeButton.style.border = 'none';
            closeButton.style.borderRadius = '3px';
            closeButton.style.cursor = 'pointer';

            closeButton.onclick = function() {{
                var resultDiv = document.getElementById('resultDiv');
                resultDiv.parentNode.removeChild(resultDiv);
            }};
            """
            crawler.driver.execute_script(js_code_display_second_popup)

            # 移除触发元素，确保可以再次点击触发
            trigger_element = crawler.driver.find_element(By.ID, "pythonTrigger")
            crawler.driver.execute_script("arguments[0].parentNode.removeChild(arguments[0]);", trigger_element)
        # except:
        #     pass
        
run(crawler, model)