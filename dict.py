from math import *

# 楕円体
ELLIPSOID_GRS80 = 1 # GRS80
ELLIPSOID_WGS84 = 2 # WGS84

# 楕円体ごとの長軸半径と扁平率
GEODETIC_DATUM = {
    ELLIPSOID_GRS80: [
        6378137.0,         # [GRS80]長軸半径
        1 / 298.257222101, # [GRS80]扁平率
    ],
    ELLIPSOID_WGS84: [
        6378137.0,         # [WGS84]長軸半径
        1 / 298.257223563, # [WGS84]扁平率
    ],
}

# 反復計算の上限回数
ITERATION_LIMIT = 1000

'''
Vincenty法(逆解法)
2地点の座標(緯度経度)から、距離と方位角を計算する
:param lat1: 始点の緯度
:param lon1: 始点の経度
:param lat2: 終点の緯度
:param lon2: 終点の経度
:param ellipsoid: 楕円体
:return: 距離と方位角
'''
def vincenty_inverse(lat1, lon1, lat2, lon2, ellipsoid=None):

    # 差異が無ければ0.0を返す
    if isclose(lat1, lat2) and isclose(lon1, lon2):
        return {
            'distance': 0.0,
            'azimuth1': 0.0,
            'azimuth2': 0.0,
        }

    # 計算時に必要な長軸半径(a)と扁平率(ƒ)を定数から取得し、短軸半径(b)を算出する
    # 楕円体が未指定の場合はGRS80の値を用いる
    a, ƒ = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))
    b = (1 - ƒ) * a

    φ1 = radians(lat1)
    φ2 = radians(lat2)
    λ1 = radians(lon1)
    λ2 = radians(lon2)

    # 更成緯度(補助球上の緯度)
    U1 = atan((1 - ƒ) * tan(φ1))
    U2 = atan((1 - ƒ) * tan(φ2))

    sinU1 = sin(U1)
    sinU2 = sin(U2)
    cosU1 = cos(U1)
    cosU2 = cos(U2)

    # 2点間の経度差
    L = λ2 - λ1

    # λをLで初期化
    λ = L

    # 以下の計算をλが収束するまで反復する
    # 地点によっては収束しないことがあり得るため、反復回数に上限を設ける
    for i in range(ITERATION_LIMIT):
        sinλ = sin(λ)
        cosλ = cos(λ)
        sinσ = sqrt((cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)
        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = atan2(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        cos2α = 1 - sinα ** 2
        cos2σm = cosσ - 2 * sinU1 * sinU2 / cos2α
        C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))
        λʹ = λ
        λ = L + (1 - C) * ƒ * sinα * (σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))

        # 偏差が.000000000001以下ならbreak
        if abs(λ - λʹ) <= 1e-12:
            break
    else:
        # 計算が収束しなかった場合はNoneを返す
        return None

    # λが所望の精度まで収束したら以下の計算を行う
    u2 = cos2α * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-1 + 2 * cos2σm ** 2) - B / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)))

    # 2点間の楕円体上の距離
    s = b * A * (σ - Δσ)

    # 各点における方位角
    α1 = atan2(cosU2 * sinλ, cosU1 * sinU2 - sinU1 * cosU2 * cosλ)
    α2 = atan2(cosU1 * sinλ, -sinU1 * cosU2 + cosU1 * sinU2 * cosλ) + pi

    if α1 < 0:
        α1 = α1 + pi * 2

    return {
        'distance': s,           # 距離
        'azimuth1': degrees(α1), # 方位角(始点→終点)
        'azimuth2': degrees(α2), # 方位角(終点→始点)
    }

lat1 = 38.21802  # 仙台市体育館の緯度
lon1 = 140.86918 # 仙台市体育館の経度
lat2 = 38.26033  # 仙台駅の緯度
lon2 = 140.88243 # 仙台駅の経度

result = vincenty_inverse(lat1, lon1, lat2, lon2, 1)

print('距離：%s(m)' % round(result['distance'], 3))
print('方位角(始点→終点)：%s' % result['azimuth1'])
print('方位角(終点→始点)：%s' % result['azimuth2'])