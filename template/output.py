"""
{
  "original_video_id": "v00001.mp4",                     //原始视频的唯一标识符
  "video_title": "",
  "source_url": "https://example.com/video/path",       //是原始视频源地址
  "clip_id": "c00001.mp4",                              //视频片段在原始视频中拆分出的顺序编号
  "filename": " v00001_c00001.mp4",                     //视频片段存储名字
  "start_time": "00:01:23.450",                         //片段开始具体时间戳
  "end_time": "00:02:10.590",                           //片段结束具体时间戳
  "duration_seconds": 47.14,                            //视频片段总时长
  "speaking_duration": 40,                              //说话人在片段中讲话的精确时长
  "transcript": "这是片段的转录文本（如果可用）",
  "speaker_upper_body_visible": true,                   //字段表示说话人的上半身是否在整个片段中始终可见。
  "topic_keywords": ["关键词1", "关键词2", "关键词3"],     //包含了爬取时所依据的关键词列表
  "speaker_full_body_visible": false,                   //字段则表示说话人的全身是否在片段中可见
  "category": "新闻播报",                                //人物属性或场景类型分类
  "speaker_name": "诸葛亮",                              //记录说话人的姓名,如果可知
  "language": ""
}
"""