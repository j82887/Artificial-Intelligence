opencv_traincascade.exe -data cascades -vec vector/facevector.vec -bg negative/bg.txt -numPos 497 -numNeg 293 -numStages 15 -w 76 -h 20 -minHitRate 0.999 -precalcValBufSize 512 -precalcIdxBufSize 512 -mode ALL
