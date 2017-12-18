#!/bin/bash
basePath="/aifs1/users/kxd/sre/data/test/xytx_far_1214/"
destPath="/aifs1/users/lj/voice/data/xytx_far_1214/test/audio/"
file=$basePath"test/wav.scp"
length=`cat $file | wc -l`
echo $file 
echo $length
index=1
while(($index<=$length))
do
    audioFile=`awk '{print $2}' $file | head -n $index | tail -n -1`
    audioName=${audioFile##*/}
    dirName=${audioFile%/*}
    echo $audioName $dirName
    let "index++"
    destFile=$destPath$audioName
    #echo $destFile

    scale=`sox $audioFile -n stat -v 2>&1`
    scale=`echo "$scale*0.9"|bc`
    sox -v $scale $audioFile -r 16000 -b 16 -c 1 $destFile
    #break
done
: '
for file in split6/*.wav; do
    scale=`sox $file -n stat -v 2>&1`
    echo $scale $file
    scale=`echo "$scale*0.9"|bc`
    newfile=${file//split6/split6_}
    echo $file $newfile output:$scale
    sox -v $scale $file -r 16000 -b 16 -c 1 $newfile
    #rm -rf $c
    #mv new_$c $c
    break
done
'
