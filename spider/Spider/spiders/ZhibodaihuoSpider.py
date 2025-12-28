# # -*- coding: utf-8 -*-

# 数据爬取文件

import scrapy
import pymysql
import pymssql
from ..items import ZhibodaihuoItem
import time
from datetime import datetime,timedelta
import datetime as formattime
import re
import random
import platform
import json
import os
import urllib
from urllib.parse import urlparse
import requests
import emoji
import numpy as np
from DrissionPage import Chromium
import pandas as pd
from sqlalchemy import create_engine
from selenium.webdriver import ChromeOptions, ActionChains
from scrapy.http import TextResponse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import pandas as pd
from sqlalchemy import create_engine
from selenium.webdriver import ChromeOptions, ActionChains
from scrapy.http import TextResponse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.service import Service
# 直播带货
class ZhibodaihuoSpider(scrapy.Spider):
    name = 'zhibodaihuoSpider'
    spiderUrl = 'https://www.douyin.com/aweme/v1/web/general/search/single/?device_platform=webapp&aid=6383&channel=channel_pc_web&search_channel=aweme_general&enable_history=1&keyword=%E7%9B%B4%E6%92%AD%E5%8D%96%E8%B4%A7&search_source=normal_search&query_correct_type=1&is_filter_search=0&from_group_id=&offset=10&count=10&need_filter_settings=0&list_type=single&search_id=20250308192455BCD4185EC4DED7EA16C7&update_version_code=170400&pc_client_type=1&pc_libra_divert=Windows&support_h265=1&support_dash=1&version_code=190600&version_name=19.6.0&cookie_enabled=true&screen_width=1536&screen_height=864&browser_language=zh-CN&browser_platform=Win32&browser_name=Chrome&browser_version=133.0.0.0&browser_online=true&engine_name=Blink&engine_version=133.0.0.0&os_name=Windows&os_version=10&cpu_core_num=12&device_memory=8&platform=PC&downlink=10&effective_type=4g&round_trip_time=50&webid=7455591226525107775&uifid=c4a29131752d59acb78af076c3dbdd52744118e38e80b4b96439ef1e20799db0577483dda0427b26dcb628c565c3faf2ade9f8a4e47fa9daaa7fb53178ef5f9cc7afb471e951792bfc8a6954f24b84cedce792c2521824337d4c4078ac9f4c067a2abdd0015d6dfea517ffc87c250bec73fd520fc8469f778177f8a4ac117bd149e7567e018ba59d2e3386989035e174de351725aceb6eba2b294c8bfd859d54&msToken=C09EEkIAC5vhF66nG5zobQnUaeW7CJriru-64EgVD4M4eNxnG-k27Q4O3DP5SoywAz8qI77Io9RbGCL1TzcHPlOFFFT0Jlcx8XE8mKL8Rbx859cGP2qKx7CuwZIM5OMRB_hdwMEX4T8LgkKHlAxb6b84wUs6o3T4zg2mPOerKhJ7Z-V1R3U%3D&a_bogus=D74RkFtjYZ5bcdKS8OGztfBl19j%2FrPSyWPTKWJ-ltPNiTHlbTbPbNnaijoq6BcozNYBwiqI7RDM%2FYDdc%2FU7sZqnpKmpkSMhSizQVVWso8H71aGJg9ZRwSvGxLi-TWSGPO5AGEri1A0Uw1g5fNHniloP9CAeEB%2FR8sqaRpPWUSxgQ64kYVVV1CPZT'
    start_urls = spiderUrl.split(";")
    protocol = ''
    hostname = ''
    realtime = False


    def __init__(self,realtime=False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.realtime = realtime=='true'

    def start_requests(self):

        plat = platform.system().lower()
        if not self.realtime and (plat == 'linux' or plat == 'windows'):
            connect = self.db_connect()
            cursor = connect.cursor()
            if self.table_exists(cursor, '62ps8694_zhibodaihuo') == 1:
                cursor.close()
                connect.close()
                self.temp_data()
                return
        pageNum = 1 + 1

        for url in self.start_urls:
            if '{}' in url:
                for page in range(1, pageNum):

                    next_link = url.format(page)
                    yield scrapy.Request(
                        url=next_link,
                        callback=self.parse
                    )
            else:
                yield scrapy.Request(
                    url=url,
                    callback=self.parse
                )

    # 列表解析
    def parse(self, response):
        _url = urlparse(self.spiderUrl)
        self.protocol = _url.scheme
        self.hostname = _url.netloc
        plat = platform.system().lower()
        if not self.realtime and (plat == 'linux' or plat == 'windows'):
            connect = self.db_connect()
            cursor = connect.cursor()
            if self.table_exists(cursor, '62ps8694_zhibodaihuo') == 1:
                cursor.close()
                connect.close()
                self.temp_data()
                return
        data = json.loads(response.body)
        try:
            list = data["data"]
        except:
            pass
        for item in list:
            fields = ZhibodaihuoItem()


            try:
                fields["title"] = emoji.demojize(self.remove_html(str( item["aweme_info"]["desc"] )))

            except:
                pass
            try:
                fields["nickname"] = emoji.demojize(self.remove_html(str( item["aweme_info"]["author"]["nickname"] )))

            except:
                pass
            try:
                fields["imgurl"] = emoji.demojize(self.remove_html(str( item["aweme_info"]["video"]["cover"]["url_list"][0] )))

            except:
                pass
            try:
                fields["duration"] = int( item["aweme_info"]["video"]["duration"])
            except:
                pass
            try:
                fields["ratio"] = emoji.demojize(self.remove_html(str( item["aweme_info"]["video"]["ratio"] )))

            except:
                pass
            try:
                fields["collectcount"] = int( item["aweme_info"]["statistics"]["collect_count"])
            except:
                pass
            try:
                fields["commentcount"] = int( item["aweme_info"]["statistics"]["comment_count"])
            except:
                pass
            try:
                fields["diggcount"] = int( item["aweme_info"]["statistics"]["digg_count"])
            except:
                pass
            try:
                fields["sharecount"] = int( item["aweme_info"]["statistics"]["share_count"])
            except:
                pass
            try:
                fields["cjtime"] = emoji.demojize(self.remove_html(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime( item["aweme_info"]["create_time"])) )))

            except:
                pass
            try:
                fields["detailurl"] = emoji.demojize(self.remove_html(str('https://www.douyin.com/video/'+ item["aweme_info"]["aweme_id"] )))

            except:
                pass
            yield fields

    # 详情解析
    def detail_parse(self, response):
        fields = response.meta['fields']
        return fields

    # 数据清洗
    def pandas_filter(self):
        engine = create_engine('mysql+pymysql://root:123456@localhost/spider62ps8694?charset=UTF8MB4')
        df = pd.read_sql('select * from zhibodaihuo limit 50', con = engine)

        # 重复数据过滤
        df.duplicated()
        df.drop_duplicates()

        #空数据过滤
        df.isnull()
        df.dropna()

        # 填充空数据
        df.fillna(value = '暂无')

        # 异常值过滤

        # 滤出 大于800 和 小于 100 的
        a = np.random.randint(0, 1000, size = 200)
        cond = (a<=800) & (a>=100)
        a[cond]

        # 过滤正态分布的异常值
        b = np.random.randn(100000)
        # 3σ过滤异常值，σ即是标准差
        cond = np.abs(b) > 3 * 1
        b[cond]

        # 正态分布数据
        df2 = pd.DataFrame(data = np.random.randn(10000,3))
        # 3σ过滤异常值，σ即是标准差
        cond = (df2 > 3*df2.std()).any(axis = 1)
        # 不满⾜条件的⾏索引
        index = df2[cond].index
        # 根据⾏索引，进⾏数据删除
        df2.drop(labels=index,axis = 0)

    # 去除多余html标签
    def remove_html(self, html):
        if html == None:
            return ''
        pattern = re.compile(r'<[^>]+>', re.S)
        return pattern.sub('', html).strip()

    # 数据库连接
    def db_connect(self):
        type = self.settings.get('TYPE', 'mysql')
        host = self.settings.get('HOST', 'localhost')
        port = int(self.settings.get('PORT', 3306))
        user = self.settings.get('USER', 'root')
        password = self.settings.get('PASSWORD', '123456')

        try:
            database = self.databaseName
        except:
            database = self.settings.get('DATABASE', '')

        if type == 'mysql':
            connect = pymysql.connect(host=host, port=port, db=database, user=user, passwd=password, charset='utf8mb4')
        else:
            connect = pymssql.connect(host=host, user=user, password=password, database=database)
        return connect

    # 断表是否存在
    def table_exists(self, cursor, table_name):
        cursor.execute("show tables;")
        tables = [cursor.fetchall()]
        table_list = re.findall('(\'.*?\')',str(tables))
        table_list = [re.sub("'",'',each) for each in table_list]

        if table_name in table_list:
            return 1
        else:
            return 0

    # 数据缓存源
    def temp_data(self):

        connect = self.db_connect()
        cursor = connect.cursor()
        sql = '''
            insert into `zhibodaihuo`(
                id
                ,title
                ,nickname
                ,imgurl
                ,duration
                ,ratio
                ,collectcount
                ,commentcount
                ,diggcount
                ,sharecount
                ,cjtime
                ,detailurl
            )
            select
                id
                ,title
                ,nickname
                ,imgurl
                ,duration
                ,ratio
                ,collectcount
                ,commentcount
                ,diggcount
                ,sharecount
                ,cjtime
                ,detailurl
            from `62ps8694_zhibodaihuo`
            where(not exists (select
                id
                ,title
                ,nickname
                ,imgurl
                ,duration
                ,ratio
                ,collectcount
                ,commentcount
                ,diggcount
                ,sharecount
                ,cjtime
                ,detailurl
            from `zhibodaihuo` where
                `zhibodaihuo`.id=`62ps8694_zhibodaihuo`.id
            ))
            order by rand()
            limit 50;
        '''

        cursor.execute(sql)
        connect.commit()
        connect.close()
