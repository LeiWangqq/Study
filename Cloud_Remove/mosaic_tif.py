"""
@FileName：mosaic_tif.py\n
@Description：多个tif图像镶嵌生成新图像\n
@Author：Wang.Lei\n
@Time：2023/1/10 18:21\n
@Department：Postgrate\n
"""
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys

driver=webdriver.Chrome(executable_path='C:\\Users\Administrator\Anaconda3\Scripts\chromedriver.exe')

driver.get('https://www.gscloud.cn/accounts/login')

email=driver.find_element_by_xpath('//*[@id="userid"]')
email.send_keys('你自己的账号')
password=driver.find_element_by_xpath('//*[@id="password"]')
password.send_keys('你自己的密码')
captcha=driver.find_element_by_xpath('//*[@id="id_captcha_1"]')
captcha_sj=input('请输入验证码：').strip()
captcha.send_keys(captcha_sj)

dr_buttoon=driver.find_element_by_xpath('//*[@id="login-form"]/input[3]').click()
time.sleep(3)
sjzy=driver.find_element_by_xpath('/html/body/div[3]/div[3]/div[5]/a/h4').click()
time.sleep(3)
GDEMV30=driver.find_element_by_xpath('//*[@id="dataset-listview"]/div/div/ul/li[4]/div/a[3]').click()
time.sleep(3)

#一共是2261页
page_num=2261
page=1
while page<=page_num:
    print('当前下载第{}页'.format(page))
    for tr_num in range(3,13): #只能取到3-12
        d_everypage='//*[@id="all_datasets_listview"]/div/table/tbody/tr['+str(tr_num)+']/td[9]/div/div/a[2]/span'
        download=driver.find_element_by_xpath(d_everypage).click()
        time.sleep(20)  #每个下载时间给20秒
    page += 1
    page_sr=driver.find_element_by_xpath('//*[@id="pager1"]/div[2]/table/tbody/tr/td[7]/input')
    page_sr.clear()
    page_sr.send_keys(page)
    page_sr.send_keys(Keys.RETURN)
    time.sleep(3)
