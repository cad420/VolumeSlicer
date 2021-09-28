# OffScreenVolumeRenderer
## input file
使用json
```json
{
  "fps": 30,
  "width": 800,
  "height": 800,
  "output_video_name": "result.avi",
  "volume_data_config": "E:/MouseNeuronData/mouse_file_config.json",
  "space": [0.00032,0.00032,0.001],
  "lod_policy": [0.3,0.5,1.2,1.6,3.2,6.4,-1.0],
  "tf": {
    "0": [0.0,0.0,0.0,0.0],
    "25": [0.0,0.0,0.0,0.0],
    "60": [0.75,0.5,1.0,0.0],
    "64": [0.75,0.75,0.75,0.9],
    "224": [1.0,0.5,0.75,0.9],
    "255": [1.0,1.0,1.0,1.0]
  },
  "camera_sequence_config": "camera_sequence_config.json"
}
```
[zoom,pos,look_at,up,right]
```json
{
  "total_frame_count": 300,
  "property": ["zoom","pos","look_at","up","right"],
  "frame_1": [16,[5.5,6.5,6.5],[5.5,6.5,0],[0.0,1.0,0.0],[1.0,0.0,0.0]],
  "frame_2": [16,[5.5,6.5,6.5],[5.5,6.5,0],[0.0,1.0,0.0],[1.0,0.0,0.0]]
}
```