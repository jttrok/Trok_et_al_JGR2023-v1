import numpy as np
import xarray as xr

#### general analysis ####
time_period = slice("1979-01-01", "2021-12-31")
n = 15706 # number of days 
hgt_level = 500

###################################################
###################################################
################### Regions #######################
###################################################
###################################################


##### northeastern_asia input #####
northeastern_asia_input_lat_bbox = slice(59, 25)  
northeastern_asia_input_lon_bbox = slice(67, 151) 
northeastern_asia_input_t62_lats = np.array([58.0939, 56.1893, 54.2846, 52.3799, 50.4752, 48.5705, 46.6658,
       44.7611, 42.8564, 40.9517, 39.047 , 37.1422, 35.2375, 33.3328,
       31.4281, 29.5234, 27.6186, 25.7139])
northeastern_asia_input_t62_lons = np.array([ 67.5  ,  69.375,  71.25 ,  73.125,  75.   ,  76.875,  78.75 ,
        80.625,  82.5  ,  84.375,  86.25 ,  88.125,  90.   ,  91.875,
        93.75 ,  95.625,  97.5  ,  99.375, 101.25 , 103.125, 105.   ,
       106.875, 108.75 , 110.625, 112.5  , 114.375, 116.25 , 118.125,
       120.   , 121.875, 123.75 , 125.625, 127.5  , 129.375, 131.25 ,
       133.125, 135.   , 136.875, 138.75 , 140.625, 142.5  , 144.375,
       146.25 , 148.125, 150.   ])
northeastern_asia_input_nlats = len(northeastern_asia_input_t62_lats)
northeastern_asia_input_nlons = len(northeastern_asia_input_t62_lons)

##### northeastern_asia region #####
northeastern_asia_lat = slice(48, 36)
northeastern_asia_lon = slice(99, 121)
northeastern_asia_lon_EW = slice(northeastern_asia_lon.start, northeastern_asia_lon.stop)
northeastern_asia_box_y = [northeastern_asia_lat.start, northeastern_asia_lat.start, northeastern_asia_lat.stop, northeastern_asia_lat.stop, northeastern_asia_lat.start]
northeastern_asia_box_x = [northeastern_asia_lon.start, northeastern_asia_lon.stop, northeastern_asia_lon.stop, northeastern_asia_lon.start, northeastern_asia_lon.start]



###################################################
###################################################



##### southeastern_asia input #####
southeastern_asia_input_lat_bbox = slice(45, 11)  
southeastern_asia_input_lon_bbox = slice(69, 152) 
southeastern_asia_input_t62_lats = np.array([44.7611, 42.8564, 40.9517, 39.047 , 37.1422, 35.2375, 33.3328,
       31.4281, 29.5234, 27.6186, 25.7139, 23.8092, 21.9044, 19.9997,
       18.095 , 16.1902, 14.2855, 12.3808])
southeastern_asia_input_t62_lons = np.array([ 69.375,  71.25 ,  73.125,  75.   ,  76.875,  78.75 ,  80.625,
        82.5  ,  84.375,  86.25 ,  88.125,  90.   ,  91.875,  93.75 ,
        95.625,  97.5  ,  99.375, 101.25 , 103.125, 105.   , 106.875,
       108.75 , 110.625, 112.5  , 114.375, 116.25 , 118.125, 120.   ,
       121.875, 123.75 , 125.625, 127.5  , 129.375, 131.25 , 133.125,
       135.   , 136.875, 138.75 , 140.625, 142.5  , 144.375, 146.25 ,
       148.125, 150.   , 151.875])
southeastern_asia_input_nlats = len(southeastern_asia_input_t62_lats)
southeastern_asia_input_nlons = len(southeastern_asia_input_t62_lons)

##### southeastern_asia region #####
southeastern_asia_lat = slice(33, 22)
southeastern_asia_lon = slice(100, 122)
southeastern_asia_lon_EW = slice(southeastern_asia_lon.start, southeastern_asia_lon.stop)
southeastern_asia_box_y = [southeastern_asia_lat.start, southeastern_asia_lat.start, southeastern_asia_lat.stop, southeastern_asia_lat.stop, southeastern_asia_lat.start]
southeastern_asia_box_x = [southeastern_asia_lon.start, southeastern_asia_lon.stop, southeastern_asia_lon.stop, southeastern_asia_lon.start, southeastern_asia_lon.start]


###################################################
###################################################


##### southcentral_north_america input #####
southcentral_north_america_input_lat_bbox = slice(45, 11) 
southcentral_north_america_input_lon_bbox = slice(219, 303) 
southcentral_north_america_input_t62_lats = np.array([44.7611, 42.8564, 40.9517, 39.047 , 37.1422, 35.2375, 33.3328,
       31.4281, 29.5234, 27.6186, 25.7139, 23.8092, 21.9044, 19.9997,
       18.095 , 16.1902, 14.2855, 12.3808])
southcentral_north_america_input_t62_lons = np.array([219.375, 221.25 , 223.125, 225.   , 226.875, 228.75 , 230.625,
       232.5  , 234.375, 236.25 , 238.125, 240.   , 241.875, 243.75 ,
       245.625, 247.5  , 249.375, 251.25 , 253.125, 255.   , 256.875,
       258.75 , 260.625, 262.5  , 264.375, 266.25 , 268.125, 270.   ,
       271.875, 273.75 , 275.625, 277.5  , 279.375, 281.25 , 283.125,
       285.   , 286.875, 288.75 , 290.625, 292.5  , 294.375, 296.25 ,
       298.125, 300.   , 301.875])
southcentral_north_america_input_nlats = len(southcentral_north_america_input_t62_lats)
southcentral_north_america_input_nlons = len(southcentral_north_america_input_t62_lons)

##### southcentral_north_america region #####
southcentral_north_america_lat = slice(37, 21)
southcentral_north_america_lon = slice(254, 268)
southcentral_north_america_lon_EW = slice(southcentral_north_america_lon.start-360, southcentral_north_america_lon.stop-360)
southcentral_north_america_box_y = [southcentral_north_america_lat.start, southcentral_north_america_lat.start, southcentral_north_america_lat.stop, southcentral_north_america_lat.stop, southcentral_north_america_lat.start]
southcentral_north_america_box_x = [southcentral_north_america_lon.start, southcentral_north_america_lon.stop, southcentral_north_america_lon.stop, southcentral_north_america_lon.start, southcentral_north_america_lon.start]


###################################################
###################################################


##### west_texas input #####
west_texas_input_lat_bbox = slice(45, 11) 
west_texas_input_lon_bbox = slice(219, 303) 
west_texas_input_t62_lats = np.array([44.7611, 42.8564, 40.9517, 39.047 , 37.1422, 35.2375, 33.3328,
       31.4281, 29.5234, 27.6186, 25.7139, 23.8092, 21.9044, 19.9997,
       18.095 , 16.1902, 14.2855, 12.3808])
west_texas_input_t62_lons = np.array([219.375, 221.25 , 223.125, 225.   , 226.875, 228.75 , 230.625,
       232.5  , 234.375, 236.25 , 238.125, 240.   , 241.875, 243.75 ,
       245.625, 247.5  , 249.375, 251.25 , 253.125, 255.   , 256.875,
       258.75 , 260.625, 262.5  , 264.375, 266.25 , 268.125, 270.   ,
       271.875, 273.75 , 275.625, 277.5  , 279.375, 281.25 , 283.125,
       285.   , 286.875, 288.75 , 290.625, 292.5  , 294.375, 296.25 ,
       298.125, 300.   , 301.875])
west_texas_input_nlats = len(west_texas_input_t62_lats)
west_texas_input_nlons = len(west_texas_input_t62_lons)

##### west_texas region #####
west_texas_lat = slice(37, 25)
west_texas_lon = slice(254, 260)
west_texas_lon_EW = slice(west_texas_lon.start-360, west_texas_lon.stop-360)
west_texas_box_y = [west_texas_lat.start, west_texas_lat.start, west_texas_lat.stop, west_texas_lat.stop, west_texas_lat.start]
west_texas_box_x = [west_texas_lon.start, west_texas_lon.stop, west_texas_lon.stop, west_texas_lon.start, west_texas_lon.start]


###################################################
###################################################


##### east_texas input #####
east_texas_input_lat_bbox = slice(45, 11) 
east_texas_input_lon_bbox = slice(219, 303) 
east_texas_input_t62_lats = np.array([44.7611, 42.8564, 40.9517, 39.047 , 37.1422, 35.2375, 33.3328,
       31.4281, 29.5234, 27.6186, 25.7139, 23.8092, 21.9044, 19.9997,
       18.095 , 16.1902, 14.2855, 12.3808])
east_texas_input_t62_lons = np.array([219.375, 221.25 , 223.125, 225.   , 226.875, 228.75 , 230.625,
       232.5  , 234.375, 236.25 , 238.125, 240.   , 241.875, 243.75 ,
       245.625, 247.5  , 249.375, 251.25 , 253.125, 255.   , 256.875,
       258.75 , 260.625, 262.5  , 264.375, 266.25 , 268.125, 270.   ,
       271.875, 273.75 , 275.625, 277.5  , 279.375, 281.25 , 283.125,
       285.   , 286.875, 288.75 , 290.625, 292.5  , 294.375, 296.25 ,
       298.125, 300.   , 301.875])
east_texas_input_nlats = len(east_texas_input_t62_lats)
east_texas_input_nlons = len(east_texas_input_t62_lons)

##### east_texas region #####
east_texas_lat = slice(37, 26)
east_texas_lon = slice(260, 268)
east_texas_lon_EW = slice(east_texas_lon.start-360, east_texas_lon.stop-360)
east_texas_box_y = [east_texas_lat.start, east_texas_lat.start, east_texas_lat.stop, east_texas_lat.stop, east_texas_lat.start]
east_texas_box_x = [east_texas_lon.start, east_texas_lon.stop, east_texas_lon.stop, east_texas_lon.start, east_texas_lon.start]


###################################################
###################################################

##### northcentral_north_america input #####
northcentral_north_america_input_lat_bbox = slice(55, 20)  
northcentral_north_america_input_lon_bbox = slice(220, 305) 
northcentral_north_america_input_t62_lats = np.array([54.2846, 52.3799, 
                                   50.4752, 48.5705, 46.6658, 44.7611, 42.8564, 40.9517,
                                   39.047 , 37.1422, 35.2375, 33.3328, 31.4281, 29.5234, 
                                   27.6186, 25.7139, 23.8092, 21.9044])
northcentral_north_america_input_t62_lons = np.array([221.25 , 223.125, 225.   , 226.875, 228.75 , 230.625, 232.5  , 234.375,
       236.25 , 238.125, 240.   , 241.875, 243.75 , 245.625, 247.5  , 249.375,
       251.25 , 253.125, 255.   , 256.875, 258.75 , 260.625, 262.5  , 264.375,
       266.25 , 268.125, 270.   , 271.875, 273.75 , 275.625, 277.5  , 279.375,
       281.25 , 283.125, 285.   , 286.875, 288.75 , 290.625, 292.5  , 294.375,
       296.25 , 298.125, 300.   , 301.875, 303.75 ])
northcentral_north_america_input_nlats = len(northcentral_north_america_input_t62_lats)
northcentral_north_america_input_nlons = len(northcentral_north_america_input_t62_lons)

##### northcentral_north_america region #####
northcentral_north_america_lat = slice(49, 38)
northcentral_north_america_lon = slice(256, 274)
northcentral_north_america_lon_EW = slice(northcentral_north_america_lon.start-360, northcentral_north_america_lon.stop-360)
northcentral_north_america_box_y = [northcentral_north_america_lat.start, northcentral_north_america_lat.start, northcentral_north_america_lat.stop, northcentral_north_america_lat.stop, northcentral_north_america_lat.start]
northcentral_north_america_box_x = [northcentral_north_america_lon.start, northcentral_north_america_lon.stop, northcentral_north_america_lon.stop, northcentral_north_america_lon.start, northcentral_north_america_lon.start]


###################################################
###################################################


##### southeastern_north_america input #####
southeastern_north_america_input_lat_bbox = slice(50, 15)  
southeastern_north_america_input_lon_bbox = slice(229, 314) 
southeastern_north_america_input_t62_lats = np.array([48.5705, 46.6658, 44.7611, 42.8564, 40.9517, 39.047 , 37.1422,
       35.2375, 33.3328, 31.4281, 29.5234, 27.6186, 25.7139, 23.8092,
       21.9044, 19.9997, 18.095 , 16.1902])
southeastern_north_america_input_t62_lons = np.array([230.625, 232.5  , 234.375, 236.25 , 238.125, 240.   , 241.875,
       243.75 , 245.625, 247.5  , 249.375, 251.25 , 253.125, 255.   ,
       256.875, 258.75 , 260.625, 262.5  , 264.375, 266.25 , 268.125,
       270.   , 271.875, 273.75 , 275.625, 277.5  , 279.375, 281.25 ,
       283.125, 285.   , 286.875, 288.75 , 290.625, 292.5  , 294.375,
       296.25 , 298.125, 300.   , 301.875, 303.75 , 305.625, 307.5  ,
       309.375, 311.25 , 313.125])
southeastern_north_america_input_nlats = len(southeastern_north_america_input_t62_lats)
southeastern_north_america_input_nlons = len(southeastern_north_america_input_t62_lons)

##### southeastern_north_america region #####
southeastern_north_america_lat = slice(37, 25)
southeastern_north_america_lon = slice(268, 285)
southeastern_north_america_lon_EW = slice(southeastern_north_america_lon.start-360, southeastern_north_america_lon.stop-360)
southeastern_north_america_box_y = [southeastern_north_america_lat.start, southeastern_north_america_lat.start, southeastern_north_america_lat.stop, southeastern_north_america_lat.stop, southeastern_north_america_lat.start]
southeastern_north_america_box_x = [southeastern_north_america_lon.start, southeastern_north_america_lon.stop, southeastern_north_america_lon.stop, southeastern_north_america_lon.start, southeastern_north_america_lon.start]


###################################################
###################################################


##### southwestern_europe input #####
southwestern_europe_input_lat_bbox = slice(57, 22)  
southwestern_europe_input_lon_bbox = [slice(313, 360),slice(0,37)] 
southwestern_europe_input_t62_lats = np.array([56.1893, 54.2846, 52.3799, 50.4752, 48.5705, 46.6658, 44.7611, 42.8564,
       40.9517, 39.047 , 37.1422, 35.2375, 33.3328, 31.4281, 29.5234, 27.6186,
       25.7139, 23.8092])
southwestern_europe_input_t62_lons = np.array([313.125, 315.   , 316.875, 318.75 , 320.625, 322.5  , 324.375, 326.25 ,
       328.125, 330.   , 331.875, 333.75 , 335.625, 337.5  , 339.375, 341.25 ,
       343.125, 345.   , 346.875, 348.75 , 350.625, 352.5  , 354.375, 356.25 ,
       358.125,   0.   ,   1.875,   3.75 ,   5.625,   7.5  ,   9.375,  11.25 ,
        13.125,  15.   ,  16.875,  18.75 ,  20.625,  22.5  ,  24.375,  26.25 ,
        28.125,  30.   ,  31.875,  33.75 ,  35.625])
southwestern_europe_input_nlats = len(southwestern_europe_input_t62_lats)
southwestern_europe_input_nlons = len(southwestern_europe_input_t62_lons)

##### southwestern_europe region #####
southwestern_europe_lat = slice(43, 36)
southwestern_europe_lon = [slice(350, 360),slice(0,1)]
southwestern_europe_lon_EW = slice(southwestern_europe_lon[0].start-360, southwestern_europe_lon[1].stop)
southwestern_europe_box_y = [southwestern_europe_lat.start, southwestern_europe_lat.start, southwestern_europe_lat.stop, southwestern_europe_lat.stop, southwestern_europe_lat.start]
southwestern_europe_box_x = [southwestern_europe_lon[0].start, southwestern_europe_lon[1].stop, southwestern_europe_lon[1].stop, southwestern_europe_lon[0].start, southwestern_europe_lon[0].start]


###################################################
###################################################


##### western_europe input #####
western_europe_input_lat_bbox = slice(63, 29)  
western_europe_input_lon_bbox = [slice(318, 360),slice(0,42)] 
western_europe_input_t62_lats = np.array([61.9033, 59.9986, 58.0939, 56.1893, 54.2846, 52.3799, 50.4752, 48.5705,
       46.6658, 44.7611, 42.8564, 40.9517, 39.047 , 37.1422, 35.2375, 33.3328,
       31.4281, 29.5234])
western_europe_input_t62_lons = np.array([318.75 , 320.625, 322.5  , 324.375, 326.25 , 328.125, 330.   , 331.875,
       333.75 , 335.625, 337.5  , 339.375, 341.25 , 343.125, 345.   , 346.875,
       348.75 , 350.625, 352.5  , 354.375, 356.25 , 358.125,   0.   ,   1.875,
         3.75 ,   5.625,   7.5  ,   9.375,  11.25 ,  13.125,  15.   ,  16.875,
        18.75 ,  20.625,  22.5  ,  24.375,  26.25 ,  28.125,  30.   ,  31.875,
        33.75 ,  35.625,  37.5  ,  39.375,  41.25 ])
western_europe_input_nlats = len(western_europe_input_t62_lats)
western_europe_input_nlons = len(western_europe_input_t62_lons)

##### western_europe region #####
western_europe_lat = slice(50, 43)
western_europe_lon = [slice(355, 360),slice(0,6)]
western_europe_lon_EW = slice(western_europe_lon[0].start-360, western_europe_lon[1].stop)
western_europe_box_y = [western_europe_lat.start, western_europe_lat.start, western_europe_lat.stop, western_europe_lat.stop, western_europe_lat.start]
western_europe_box_x = [western_europe_lon[0].start, western_europe_lon[1].stop, western_europe_lon[1].stop, western_europe_lon[0].start, western_europe_lon[0].start]


###################################################
###################################################


##### central_europe input #####
central_europe_input_lat_bbox = slice(69, 34)  
central_europe_input_lon_bbox = [slice(331, 360),slice(0,55)]
central_europe_input_t62_lats = np.array([67.6171, 65.7125, 63.8079, 61.9033, 59.9986, 58.0939, 56.1893, 54.2846,
       52.3799, 50.4752, 48.5705, 46.6658, 44.7611, 42.8564, 40.9517, 39.047 ,
       37.1422, 35.2375])
central_europe_input_t62_lons = np.array([331.875, 333.75 , 335.625, 337.5  , 339.375, 341.25 , 343.125, 345.   ,
       346.875, 348.75 , 350.625, 352.5  , 354.375, 356.25 , 358.125,   0.   ,
         1.875,   3.75 ,   5.625,   7.5  ,   9.375,  11.25 ,  13.125,  15.   ,
        16.875,  18.75 ,  20.625,  22.5  ,  24.375,  26.25 ,  28.125,  30.   ,
        31.875,  33.75 ,  35.625,  37.5  ,  39.375,  41.25 ,  43.125,  45.   ,
        46.875,  48.75 ,  50.625,  52.5  ,  54.375])
central_europe_input_nlats = len(central_europe_input_t62_lats)
central_europe_input_nlons = len(central_europe_input_t62_lons)

##### central_europe region #####
central_europe_lat = slice(55, 48)
central_europe_lon = slice(6,19)
central_europe_lon_EW = central_europe_lon
central_europe_box_y = [central_europe_lat.start, central_europe_lat.start, central_europe_lat.stop, central_europe_lat.stop, central_europe_lat.start]
central_europe_box_x = [central_europe_lon.start, central_europe_lon.stop, central_europe_lon.stop, central_europe_lon.start, central_europe_lon.start]

                          
###################################################
###################################################


##### eastern_europe input #####
eastern_europe_input_lat_bbox = slice(62, 28)  
eastern_europe_input_lon_bbox = [slice(330, 360),slice(0,53)]
eastern_europe_input_t62_lats = np.array([61.9033, 59.9986, 58.0939, 56.1893, 54.2846, 52.3799, 50.4752, 48.5705,
       46.6658, 44.7611, 42.8564, 40.9517, 39.047 , 37.1422, 35.2375, 33.3328,
       31.4281, 29.5234])
eastern_europe_input_t62_lons = np.array([330.   , 331.875, 333.75 , 335.625, 337.5  , 339.375, 341.25 , 343.125,
       345.   , 346.875, 348.75 , 350.625, 352.5  , 354.375, 356.25 , 358.125,
         0.   ,   1.875,   3.75 ,   5.625,   7.5  ,   9.375,  11.25 ,  13.125,
        15.   ,  16.875,  18.75 ,  20.625,  22.5  ,  24.375,  26.25 ,  28.125,
        30.   ,  31.875,  33.75 ,  35.625,  37.5  ,  39.375,  41.25 ,  43.125,
        45.   ,  46.875,  48.75 ,  50.625,  52.5  ])
eastern_europe_input_nlats = len(eastern_europe_input_t62_lats)
eastern_europe_input_nlons = len(eastern_europe_input_t62_lons)

##### eastern_europe region #####
eastern_europe_lat = slice(48, 41)
eastern_europe_lon = slice(17, 29)
eastern_europe_lon_EW = eastern_europe_lon
eastern_europe_box_y = [eastern_europe_lat.start, eastern_europe_lat.start, eastern_europe_lat.stop, eastern_europe_lat.stop, eastern_europe_lat.start]
eastern_europe_box_x = [eastern_europe_lon.start, eastern_europe_lon.stop, eastern_europe_lon.stop, eastern_europe_lon.start, eastern_europe_lon.start]


###################################################
###################################################


##### northeastern_europe input #####
northeastern_europe_input_lat_bbox = slice(71, 36)  
northeastern_europe_input_lon_bbox = slice(3,87)
northeastern_europe_input_t62_lats = np.array([69.5217, 67.6171, 65.7125, 63.8079, 61.9033, 59.9986, 58.0939, 56.1893,
       54.2846, 52.3799, 50.4752, 48.5705, 46.6658, 44.7611, 42.8564, 40.9517,
       39.047 , 37.1422])
northeastern_europe_input_t62_lons = np.array([ 3.75 ,  5.625,  7.5  ,  9.375, 11.25 , 13.125, 15.   , 16.875, 18.75 ,
       20.625, 22.5  , 24.375, 26.25 , 28.125, 30.   , 31.875, 33.75 , 35.625,
       37.5  , 39.375, 41.25 , 43.125, 45.   , 46.875, 48.75 , 50.625, 52.5  ,
       54.375, 56.25 , 58.125, 60.   , 61.875, 63.75 , 65.625, 67.5  , 69.375,
       71.25 , 73.125, 75.   , 76.875, 78.75 , 80.625, 82.5  , 84.375, 86.25 ])
northeastern_europe_input_nlats = len(northeastern_europe_input_t62_lats)
northeastern_europe_input_nlons = len(northeastern_europe_input_t62_lons)

##### northeastern_europe region #####
northeastern_europe_lat = slice(59, 51)
northeastern_europe_lon = slice(37, 53)
northeastern_europe_lon_EW = northeastern_europe_lon
northeastern_europe_box_y = [northeastern_europe_lat.start, northeastern_europe_lat.start, northeastern_europe_lat.stop, northeastern_europe_lat.stop, northeastern_europe_lat.start]
northeastern_europe_box_x = [northeastern_europe_lon.start, northeastern_europe_lon.stop, northeastern_europe_lon.stop, northeastern_europe_lon.start, northeastern_europe_lon.start]


###################################################
###################################################


##### southwestern_australia input #####
southwestern_australia_input_lat_bbox = slice(-14, -47)  
southwestern_australia_input_lon_bbox = slice(80,164)
southwestern_australia_input_t62_lats = np.array([-14.2855, -16.1902, -18.095 , -19.9997, -21.9044, -23.8092, -25.7139,
       -27.6186, -29.5234, -31.4281, -33.3328, -35.2375, -37.1422, -39.047 ,
       -40.9517, -42.8564, -44.7611, -46.6658])
southwestern_australia_input_t62_lons = np.array([80.625,  82.5  ,  84.375,  86.25 ,  88.125,  90.   ,  91.875,  93.75 ,
        95.625,  97.5  ,  99.375, 101.25 , 103.125, 105.   , 106.875, 108.75 ,
       110.625, 112.5  , 114.375, 116.25 , 118.125, 120.   , 121.875, 123.75 ,
       125.625, 127.5  , 129.375, 131.25 , 133.125, 135.   , 136.875, 138.75 ,
       140.625, 142.5  , 144.375, 146.25 , 148.125, 150.   , 151.875, 153.75 ,
       155.625, 157.5  , 159.375, 161.25 , 163.125])
southwestern_australia_input_nlats = len(southwestern_australia_input_t62_lats)
southwestern_australia_input_nlons = len(southwestern_australia_input_t62_lons)

##### southwestern_australia region #####
southwestern_australia_lat = slice(-25, -36)
southwestern_australia_lon = slice(112,133)
southwestern_australia_lon_EW = southwestern_australia_lon
southwestern_australia_box_y = [southwestern_australia_lat.start, southwestern_australia_lat.start, southwestern_australia_lat.stop, southwestern_australia_lat.stop, southwestern_australia_lat.start]
southwestern_australia_box_x = [southwestern_australia_lon.start, southwestern_australia_lon.stop, southwestern_australia_lon.stop, southwestern_australia_lon.start, southwestern_australia_lon.start]


###################################################
###################################################

##### southeastern_australia input #####
southeastern_australia_input_lat_bbox = slice(-15, -50)  
southeastern_australia_input_lon_bbox = slice(103,186)
southeastern_australia_input_t62_lats = np.array([-16.1902, -18.095 , -19.9997, -21.9044, -23.8092, -25.7139, -27.6186,
       -29.5234, -31.4281, -33.3328, -35.2375, -37.1422, -39.047 , -40.9517,
       -42.8564, -44.7611, -46.6658, -48.5705])
southeastern_australia_input_t62_lons = np.array([103.125, 105.   , 106.875, 108.75 , 110.625, 112.5  , 114.375, 116.25 ,
       118.125, 120.   , 121.875, 123.75 , 125.625, 127.5  , 129.375, 131.25 ,
       133.125, 135.   , 136.875, 138.75 , 140.625, 142.5  , 144.375, 146.25 ,
       148.125, 150.   , 151.875, 153.75 , 155.625, 157.5  , 159.375, 161.25 ,
       163.125, 165.   , 166.875, 168.75 , 170.625, 172.5  , 174.375, 176.25 ,
       178.125, 180.   , 181.875, 183.75 , 185.625])
southeastern_australia_input_nlats = len(southeastern_australia_input_t62_lats)
southeastern_australia_input_nlons = len(southeastern_australia_input_t62_lons)

##### southeastern_australia region #####
southeastern_australia_lat = slice(-27, -39)
southeastern_australia_lon = slice(135,154)
southeastern_australia_lon_EW = southeastern_australia_lon
southeastern_australia_box_y = [southeastern_australia_lat.start, southeastern_australia_lat.start, southeastern_australia_lat.stop, southeastern_australia_lat.stop, southeastern_australia_lat.start]
southeastern_australia_box_x = [southeastern_australia_lon.start, southeastern_australia_lon.stop, southeastern_australia_lon.stop, southeastern_australia_lon.start, southeastern_australia_lon.start]


###################################################
###################################################

##### southeastern_africa input #####
southeastern_africa_input_lat_bbox = slice(-11, -45)  
southeastern_africa_input_lon_bbox = [slice(348, 360), slice(0,72)]
southeastern_africa_input_t62_lats = np.array([-12.3808, -14.2855, -16.1902, -18.095 , -19.9997, -21.9044, -23.8092,
       -25.7139, -27.6186, -29.5234, -31.4281, -33.3328, -35.2375, -37.1422,
       -39.047 , -40.9517, -42.8564, -44.7611])
southeastern_africa_input_t62_lons = np.array([348.75 , 350.625, 352.5  , 354.375, 356.25 , 358.125,   0.   ,   1.875,
         3.75 ,   5.625,   7.5  ,   9.375,  11.25 ,  13.125,  15.   ,  16.875,
        18.75 ,  20.625,  22.5  ,  24.375,  26.25 ,  28.125,  30.   ,  31.875,
        33.75 ,  35.625,  37.5  ,  39.375,  41.25 ,  43.125,  45.   ,  46.875,
        48.75 ,  50.625,  52.5  ,  54.375,  56.25 ,  58.125,  60.   ,  61.875,
        63.75 ,  65.625,  67.5  ,  69.375,  71.25])
southeastern_africa_input_nlats = len(southeastern_africa_input_t62_lats)
southeastern_africa_input_nlons = len(southeastern_africa_input_t62_lons)

##### southeastern_africa region #####
southeastern_africa_lat = slice(-20, -35)
southeastern_africa_lon = slice(25, 36)
southeastern_africa_lon_EW = southeastern_africa_lon
southeastern_africa_box_y = [southeastern_africa_lat.start, southeastern_africa_lat.start, southeastern_africa_lat.stop, southeastern_africa_lat.stop, southeastern_africa_lat.start]
southeastern_africa_box_x = [southeastern_africa_lon.start, southeastern_africa_lon.stop, southeastern_africa_lon.stop, southeastern_africa_lon.start, southeastern_africa_lon.start]


###################################################
###################################################

##### southwestern_africa input #####
southwestern_africa_input_lat_bbox = slice(-11, -45)  
southwestern_africa_input_lon_bbox = [slice(337, 360), slice(0,60)]
southwestern_africa_input_t62_lats = np.array([-12.3808, -14.2855, -16.1902, -18.095 , -19.9997, -21.9044, -23.8092,
       -25.7139, -27.6186, -29.5234, -31.4281, -33.3328, -35.2375, -37.1422,
       -39.047 , -40.9517, -42.8564, -44.7611])
southwestern_africa_input_t62_lons = np.array([337.5  , 339.375, 341.25 , 343.125, 345.   , 346.875, 348.75 , 350.625,
       352.5  , 354.375, 356.25 , 358.125,   0.   ,   1.875,   3.75 ,   5.625,
         7.5  ,   9.375,  11.25 ,  13.125,  15.   ,  16.875,  18.75 ,  20.625,
        22.5  ,  24.375,  26.25 ,  28.125,  30.   ,  31.875,  33.75 ,  35.625,
        37.5  ,  39.375,  41.25 ,  43.125,  45.   ,  46.875,  48.75 ,  50.625,
        52.5  ,  54.375,  56.25 ,  58.125,  60.])
southwestern_africa_input_nlats = len(southwestern_africa_input_t62_lats)
southwestern_africa_input_nlons = len(southwestern_africa_input_t62_lons)

##### southwestern_africa region #####
southwestern_africa_lat = slice(-20, -35)
southwestern_africa_lon = slice(12, 25)
southwestern_africa_lon_EW = southwestern_africa_lon
southwestern_africa_box_y = [southwestern_africa_lat.start, southwestern_africa_lat.start, southwestern_africa_lat.stop, southwestern_africa_lat.stop, southwestern_africa_lat.start]
southwestern_africa_box_x = [southwestern_africa_lon.start, southwestern_africa_lon.stop, southwestern_africa_lon.stop, southwestern_africa_lon.start, southwestern_africa_lon.start]



###################################################
###################################################

##### southsouthern_south_america input #####
southsouthern_south_america_input_lat_bbox = slice(-32, -67)  
southsouthern_south_america_input_lon_bbox = slice(249, 333)
southsouthern_south_america_input_t62_lats = np.array([-33.3328, -35.2375, -37.1422, -39.047 , -40.9517, -42.8564, -44.7611,
       -46.6658, -48.5705, -50.4752, -52.3799, -54.2846, -56.1893, -58.0939,
       -59.9986, -61.9033, -63.8079, -65.7125])
southsouthern_south_america_input_t62_lons = np.array([249.375, 251.25 , 253.125, 255.   , 256.875, 258.75 , 260.625, 262.5  ,
       264.375, 266.25 , 268.125, 270.   , 271.875, 273.75 , 275.625, 277.5  ,
       279.375, 281.25 , 283.125, 285.   , 286.875, 288.75 , 290.625, 292.5  ,
       294.375, 296.25 , 298.125, 300.   , 301.875, 303.75 , 305.625, 307.5  ,
       309.375, 311.25 , 313.125, 315.   , 316.875, 318.75 , 320.625, 322.5  ,
       324.375, 326.25 , 328.125, 330.   , 331.875])
southsouthern_south_america_input_nlats = len(southsouthern_south_america_input_t62_lats)
southsouthern_south_america_input_nlons = len(southsouthern_south_america_input_t62_lons)

##### southsouthern_south_america region #####
southsouthern_south_america_lat = slice(-41, -55)
southsouthern_south_america_lon = slice(284, 297)
southsouthern_south_america_lon_EW = slice(southsouthern_south_america_lon.start-360, southsouthern_south_america_lon.stop-360)
southsouthern_south_america_box_y = [southsouthern_south_america_lat.start, southsouthern_south_america_lat.start, southsouthern_south_america_lat.stop, southsouthern_south_america_lat.stop, southsouthern_south_america_lat.start]
southsouthern_south_america_box_x = [southsouthern_south_america_lon.start, southsouthern_south_america_lon.stop, southsouthern_south_america_lon.stop, southsouthern_south_america_lon.start, southsouthern_south_america_lon.start]



###################################################
###################################################

##### northsouthern_south_america input #####
northsouthern_south_america_input_lat_bbox = slice(-19, -53)  
northsouthern_south_america_input_lon_bbox = slice(256, 341)
northsouthern_south_america_input_t62_lats = np.array([-19.9997, -21.9044, -23.8092, -25.7139, -27.6186, -29.5234, -31.4281,
       -33.3328, -35.2375, -37.1422, -39.047 , -40.9517, -42.8564, -44.7611,
       -46.6658, -48.5705, -50.4752, -52.3799])
northsouthern_south_america_input_t62_lons = np.array([256.875, 258.75 , 260.625, 262.5  , 264.375, 266.25 , 268.125, 270.   ,
       271.875, 273.75 , 275.625, 277.5  , 279.375, 281.25 , 283.125, 285.   ,
       286.875, 288.75 , 290.625, 292.5  , 294.375, 296.25 , 298.125, 300.   ,
       301.875, 303.75 , 305.625, 307.5  , 309.375, 311.25 , 313.125, 315.   ,
       316.875, 318.75 , 320.625, 322.5  , 324.375, 326.25 , 328.125, 330.   ,
       331.875, 333.75 , 335.625, 337.5  , 339.375])
northsouthern_south_america_input_nlats = len(northsouthern_south_america_input_t62_lats)
northsouthern_south_america_input_nlons = len(northsouthern_south_america_input_t62_lons)

##### northsouthern_south_america region #####
northsouthern_south_america_lat = slice(-30, -41)
northsouthern_south_america_lon = slice(287, 309)
northsouthern_south_america_lon_EW = slice(northsouthern_south_america_lon.start-360, northsouthern_south_america_lon.stop-360)
northsouthern_south_america_box_y = [northsouthern_south_america_lat.start, northsouthern_south_america_lat.start, northsouthern_south_america_lat.stop, northsouthern_south_america_lat.stop, northsouthern_south_america_lat.start]
northsouthern_south_america_box_x = [northsouthern_south_america_lon.start, northsouthern_south_america_lon.stop, northsouthern_south_america_lon.stop, northsouthern_south_america_lon.start, northsouthern_south_america_lon.start]
