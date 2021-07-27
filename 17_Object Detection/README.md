## 標註器
### LabelImg標註器
* labelImg (https://github.com/tzutalin/labelImg)
1. k張影像標註完後會產生k個標註檔(.txt/.xml)
2. 每個標註檔格式為：標註編號 中心點x 中心點y 寬w 高h，其值皆為正規化後的數值

### Haar內建標註器
* Haar feature-based cascade classifier (https://github.com/sauhaardac/haar-training)
1. k張影像標註完後會產生1個標註檔(.txt)
2. 每個標註檔格式為：影像的相對位置 標註編號 左上角x 左上角y 寬w 高h
