# 数据容器文件

import scrapy

class SpiderItem(scrapy.Item):
    pass

class ZhibodaihuoItem(scrapy.Item):
    # 标题
    title = scrapy.Field()
    # 作者昵称
    nickname = scrapy.Field()
    # 图片
    imgurl = scrapy.Field()
    # 时长
    duration = scrapy.Field()
    # 分辨率
    ratio = scrapy.Field()
    # 收藏数
    collectcount = scrapy.Field()
    # 评论数
    commentcount = scrapy.Field()
    # 点赞数
    diggcount = scrapy.Field()
    # 分享数
    sharecount = scrapy.Field()
    # 创建时间
    cjtime = scrapy.Field()
    # 详情地址
    detailurl = scrapy.Field()

