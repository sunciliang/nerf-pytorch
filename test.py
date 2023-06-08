import glob
import os, imageio
from matplotlib import pyplot as plt
import numpy as np
import math
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
import rawpy
import copy

import numpy as np
# for i in range(16):
#     key = np.random.randint(i*25, (i+1)*25)
#     print(key)
# tt = []
# # # # key = [5000,5000,5000,5000,5000]
# # # # key = [2.0,1.0,0.5,0.25,0.125]
# # key = [1.0,1.0,1.0,1.0,1.0]
# key = [0,0,0,0,0]
# for i in range(16):
#     tt.append(key[i%5])
# # np.save('temperature.npy',tt)
# np.save('exposure.npy',tt)
# #
ttt = []
# key=[3200,5200,7000]
key=[1,3,7,15,31,51]
# key=[-2,0,2]
# key = [0, -2.5, -1.5, 1.5, 2.5]

# tt = [2,2,3,2,2,2,2,2,1,2,
#       1,2,2,3,2,2,2,2,1,2,
#       2,2,1,2,2,2,3,2,3,2]
# tt = [1,2,2,3,2,2,1,3,2,2,
#       2,2,2,1,2,3,2,2,3,2,
#       2,2,1,2,1,3,2,2,2,2]
tt = [1,1,1,1,1,1,1,1,1,1,
      1,1,1,1,1,1,1,1,1,1,
      1,1,1,1,1,1,1,1,1,1,
      3,4,3,2,2,3,4,3,2,5,
      4,5,2,3,4,5,2,3,4,5,
      2,3,4,5,2,3,4,5,2,3]

for i in tt:
    ttt.append(key[i-1])
# np.save('temperature.npy',ttt)
# np.save('exposure.npy',ttt)
np.save('coc.npy',ttt)
k=np.load(r'E:\GitHub\nerf-py\coc.npy')
print(k)
# k2 = np.zeros((k.shape[0]*2,k.shape[-1]))
# k2 = np.hstack((k,k))
# k2 = np.vstack((k,k)) #pose
# print(k[38],k[77])
# np.save('temperature.npy',k2)
# print(k2)
# print(k2[0],k2[16])
# path = glob.glob(r'E:\GitHub\nerf-py\data\colorboard18\RAW\*.dng')
# for idx,img in enumerate(path):
#     with rawpy.imread(img) as raw:
#             rgb = raw.postprocess(gamma=(1,1), use_auto_wb=False,use_camera_wb =True, output_bps=16)
#     imageio.imsave(r'E:\GitHub\nerf-py\data\colorboard18\images\{:03d}.jpg'.format(idx), rgb)

###################################
# import json
# import re
# import exifread
# import requests
# from exifread.classes import IfdTag
#
#
# class Field(object):
#     def __init__(self, _type=str, _default=None):
#         self.type = _type
#         self.__value__ = None
#         self.default = _default
#
#     def empty(self):
#         return self.__value__ is None
#
#     @property
#     def value(self):
#         if self.__value__ is not None:
#             return self.type(self.__value__)
#
#
# class Fields(object):
#     def __init__(self):
#         self.ExifVersion = Field()  # Exif 版本
#         self.FlashPixVersion = Field()  # FlashPix 版本
#         self.ColorSpace = Field()  # 色域、色彩空间
#         self.PixelXDimension = Field()  # 图像的有效宽度
#         self.PixelYDimension = Field()  # 图像的有效高度
#         self.ComponentsConfiguration = Field()  # 图像构造
#         self.CompressedBitsPerPixel = Field()  # 压缩时每像素色彩位
#         self.MakerNote = Field()  # 制造商设置的信息
#         self.UserComment = Field()  # 用户评论
#         self.RelatedSoundFile = Field()  # 关联的声音文件
#         self.DateTimeOriginal = Field()  # 创建时间
#         self.DateTimeDigitized = Field()  # 数字化创建时间
#         self.SubsecTime = Field()  # 日期时间（秒）
#         self.SubsecTimeOriginal = Field()  # 原始日期时间（秒）
#         self.SubsecTimeDigitized = Field()  # 原始日期时间数字化（秒）
#         # self.ExposureTime = Field()  # 曝光时间
#         # self.FNumber = Field()  # 光圈值
#         self.ExposureProgram = Field()  # 曝光程序
#         self.SpectralSensitivity = Field()  # 光谱灵敏度
#         self.ISOSpeedRatings = Field()  # 感光度
#         self.OECF = Field()  # 光电转换功能
#         # self.ShutterSpeedValue = Field()  # 快门速度
#         # self.ApertureValue = Field()  # 镜头光圈
#         self.BrightnessValue = Field()  # 亮度
#         self.ExposureBiasValue = Field()  # 曝光补偿
#         self.MaxApertureValue = Field()  # 最大光圈
#         self.SubjectDistance = Field()  # 物距
#         self.MeteringMode = Field()  # 测光方式
#         self.Lightsource = Field()  # 光源
#         self.Flash = Field()  # 闪光灯
#         self.SubjectArea = Field()  # 主体区域
#         # self.FocalLength = Field()  # 焦距
#         self.FlashEnergy = Field()  # 闪光灯强度
#         self.SpatialFrequencyResponse = Field()  # 空间频率反应
#         self.FocalPlaneXResolution = Field()  # 焦距平面X轴解析度
#         self.FocalPlaneYResolution = Field()  # 焦距平面Y轴解析度
#         self.FocalPlaneResolutionUnit = Field()  # 焦距平面解析度单位
#         self.SubjectLocation = Field()  # 主体位置
#         self.ExposureIndex = Field()  # 曝光指数
#         self.SensingMethod = Field()  # 图像传感器类型
#         self.FileSource = Field()  # 源文件
#         self.SceneType = Field()  # 场景类型（1: 直接拍摄）
#         self.CFAPattern = Field()  # CFA 模式
#         self.CustomRendered = Field()  # 自定义图像处理
#         self.ExposureMode = Field()  # 曝光模式
#         self.WhiteBalance = Field()  # 白平衡（1: 自动，2: 手动）
#         self.DigitalZoomRation = Field()  # 数字变焦
#         self.FocalLengthIn35mmFilm = Field()  # 35毫米胶片焦距
#         self.SceneCaptureType = Field()  # 场景拍摄类型
#         self.GainControl = Field()  # 场景控制
#         self.Contrast = Field()  # 对比度
#         self.Saturation = Field()  # 饱和度
#         self.Sharpness = Field()  # 锐度
#         self.DeviceSettingDescription = Field()  # 设备设定描述
#         self.SubjectDistanceRange = Field()  # 主体距离范围
#         self.InteroperabilityIFDPointer = Field()  #
#         self.ImageUniqueID = Field()  # 图像唯一ID
#         self.ImageWidth = Field()  # 图像宽度
#         self.ImageHeight = Field()  # 图像高度
#         self.BitsPerSample = Field()  # 比特采样率
#         self.Compression = Field()  # 压缩方法
#         self.PhotometricInterpretation = Field()  # 像素合成
#         self.Orientation = Field()  # 拍摄方向
#         self.SamplesPerPixel = Field()  # 像素数
#         self.PlanarConfiguration = Field()  # 数据排列
#         self.YCbCrSubSampling = Field()  # 色相抽样比率
#         self.YCbCrPositioning = Field()  # 色相配置
#         self.XResolution = Field()  # X方向分辨率
#         self.YResolution = Field()  # Y方向分辨率
#         self.ResolutionUnit = Field()  # 分辨率单位
#         self.StripOffsets = Field()  # 图像资料位置
#         self.RowsPerStrip = Field()  # 每带行数
#         self.StripByteCounts = Field()  # 每压缩带比特数
#         self.JPEGInterchangeFormat = Field()  # JPEG SOI 偏移量
#         self.JPEGInterchangeFormatLength = Field()  # JPEG 比特数
#         self.TransferFunction = Field()  # 转移功能
#         self.WhitePoint = Field()  # 白点色度
#         self.PrimaryChromaticities = Field()  # 主要色度
#         self.YCbCrCoefficients = Field()  # 颜色空间转换矩阵系数
#         self.ReferenceBlackWhite = Field()  # 黑白参照值
#         self.DateTime = Field()  # 日期和时间
#         self.ImageDescription = Field()  # 图像描述、来源
#         self.Make = Field()  # 生产者
#         self.Model = Field()  # 型号
#         self.Software = Field()  # 软件
#         self.Artist = Field()  # 作者
#         self.Copyright = Field()  # 版权信息
#         self.GPSVersionID = Field()  # GPS 版本
#         self.GPSLatitudeRef = Field()  # 南北纬
#         self.GPSLatitude = Field()  # 纬度
#         self.GPSLongitudeRef = Field()  # 东西经
#         self.GPSLongitude = Field()  # 经度
#         self.GPSAltitudeRef = Field()  # 海拔参照值
#         self.GPSAltitude = Field()  # 海拔
#         self.GPSTimeStamp = Field()  # GPS 时间戳
#         self.GPSSatellites = Field()  # 测量的卫星
#         self.GPSStatus = Field()  # 接收器状态
#         self.GPSMeasureMode = Field()  # 测量模式
#         self.GPSDOP = Field()  # 测量精度
#         self.GPSSpeedRef = Field()  # 速度单位
#         self.GPSSpeed = Field()  # GPS 接收器速度
#         self.GPSTrackRef = Field()  # 移动方位参照
#         self.GPSTrack = Field()  # 移动方位
#         self.GPSImgDirectionRef = Field()  # 图像方位参照
#         self.GPSImgDirection = Field()  # 图像方位
#         self.GPSMapDatum = Field()  # 地理测量资料
#         self.GPSDestLatitudeRef = Field()  # 目标纬度参照
#         self.GPSDestLatitude = Field()  # 目标纬度
#         self.GPSDestLongitudeRef = Field()  # 目标经度参照
#         self.GPSDestLongitude = Field()  # 目标经度
#         self.GPSDestBearingRef = Field()  # 目标方位参照
#         self.GPSDestBearing = Field()  # 目标方位
#         self.GPSDestDistanceRef = Field()  # 目标距离参照
#         self.GPSDestDistance = Field()  # 目标距离
#         self.GPSProcessingMethod = Field()  # GPS 处理方法名
#         self.GPSAreaInformation = Field()  # GPS 区功能变数名
#         self.GPSDateStamp = Field()  # GPS 日期
#         self.GPSDifferential = Field()  # GPS 修正
#
#
# class PhotoInfo(object):
#     baidu_api_key = 'your baidu ak'
#     baidu_geo_api = 'http://api.map.baidu.com/geocoder/v2/?ak={0}&callback=renderReverse&location={1},{2}&output=json&pois=0&coordtype=wgs84ll'
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36 Edg/99.0.1150.30'}
#
#     def __init__(self, path):
#         self.photo_path = path
#         self.fields = Fields()
#         self.extract_info = dict()
#         self._extract()
#
#     def _extract(self):
#         """解析图片信息"""
#         with open(self.photo_path, 'rb') as f:
#             info = exifread.process_file(f)
#         for field in self.fields.__dict__:
#             for item in info:
#                 if field in item and info.get(item):
#                     self.extract_info[field] = info[item]
#
#     @staticmethod
#     def lat_lon_to_decimal(obj: IfdTag):
#         """转化经纬度"""
#
#         def inner(_values: list):
#             val = _values.pop()
#             if _values:
#                 return val.decimal() + inner(_values) / 60
#             return val.decimal()
#
#         return inner(list(reversed(obj.values)))
#
#     def longitude_and_latitude(self):
#         """提取经纬度"""
#         # 纬度
#         lat: IfdTag = self.extract_info.get('GPSLatitude')
#         # 经度
#         lon: IfdTag = self.extract_info.get('GPSLongitude')
#         if lat and lon:
#             return self.lat_lon_to_decimal(lat), self.lat_lon_to_decimal(lon)
#         return 0, 0
#
#     def location_info(self):
#         """获取定位信息"""
#         uri = self.baidu_geo_api.format(self.baidu_api_key, *self.longitude_and_latitude())
#         resp = requests.get(uri, headers=self.headers, verify=False, timeout=15)
#         tmp = re.match(r'renderReverse&&renderReverse\((.*)\)', resp.text)
#         if tmp:
#             return json.loads(tmp.group(1)).get('result')
#
#     @staticmethod
#     def address(_location: dict):
#         """照片拍摄地址"""
#         _location = _location or {}
#         country = (_location.get('addressComponent') or {}).get('country', '')
#         province = (_location.get('addressComponent') or {}).get('province', '')
#         city = (_location.get('addressComponent') or {}).get('city', '')
#         district = (_location.get('addressComponent') or {}).get('district', '')
#         address = _location.get('sematic_description', '')
#         address_lst = [country, province, city, district, address]
#         return '-'.join(address_lst)
#
#     def date_time(self):
#         """照片拍摄时间"""
#         return self.extract_info.get('DateTime')
#
#     def phone_type(self):
#         """拍摄手机型号"""
#         return self.extract_info.get('Model')
#
#     def check_msg(self):
#         _location = self.location_info()
#         address = self.address(_location)
#         _date_time = self.date_time()
#         phone = self.phone_type()
#         return f'照片使用【{phone}】手机于【{_date_time}】在【{address}】拍摄'
#
#
# if __name__ == '__main__':
#     ph = PhotoInfo(r'E:\GitHub\nerf-py\data\three\images\IMG_20221213_211916.jpg')
#     print(ph.check_msg())
