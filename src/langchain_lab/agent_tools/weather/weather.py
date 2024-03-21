import os
from os import path
from typing import cast

import pandas as pd
import requests
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

# noinspection SpellCheckingInspection
city_codes_csv_filename = path.join(path.dirname(path.abspath(__file__)), 'AMap_adcode_citycode.csv')
city_codes = pd.read_csv(city_codes_csv_filename, encoding='utf-8')


class WeatherInput(BaseModel):
    city_name: str = Field(
        description="应该是一个中文的城市名称，这个名称不应该包含省份或区县，例如陕西省西安市碑林区或陕西西安碑林，只需要西安")


@tool("weather", args_schema=WeatherInput, return_direct=True)
def weather(city_name: str) -> str:
    """在线查询中国某个城市当前的天气"""

    # noinspection SpellCheckingInspection
    row = city_codes.query(f'中文名.str.contains("{city_name}") and adcode % 100 == 0', engine='python')
    # print(row)
    real_city_name = row.iloc[0]['中文名'] if len(row) > 0 else None
    # print(real_city_name)

    if real_city_name is None:
        return f"暂不支持查询{city_name}的天气"

    if "AMAP_API_KEY" not in os.environ:
        return f"暂不支持查询{city_name}的天气，未设置AMAP_API_KEY环境变量"

    response = requests.get('https://restapi.amap.com/v3/weather/weatherInfo', {
        'key': os.environ.get('AMAP_API_KEY'),
        'city': row.iloc[0]['adcode'],
    })
    print(response)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 1 or data['status'] == '1':
            return f"为您查询，{real_city_name}当前的天气是{data['lives'][0]['weather']}，" \
                   f"气温{data['lives'][0]['temperature']}度，空气湿度{data['lives'][0]['humidity']}。"

    return f"尝试为您查询{real_city_name}的天气，接口失败，请联系管理员"


weather = cast(BaseTool, weather)
