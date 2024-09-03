# MyoRegenTrack
Evaluating the regeneration stage of each fiber in muscle tissue of HE-stained.<be>
(using [Cellpose](https://github.com/MouseLand/cellpose), [DINO](https://github.com/facebookresearch/dino))<br>
![supplementary2](https://github.com/user-attachments/assets/52f8e5cc-423f-45ca-ab71-1cd013890b8c)<br>

cell_num:7690<br>
MB:0.5200589964637902[%]<br>
MF:0.03366217368574842[%]<br>
LMT:0.11797861576962755[%]<br>
SMT:0.3283002140808339[%]<br>
recovery_score:0.3629537549619246<br>

# Quick Start

<details><summary>Check pre setup</summary>

Please check below commnd.
```
nvidia-smi
docker
```

</details>

<!------------------------------------------------------------------------>

<details><summary>Model install</summary>


```
cd model
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
cd ..
```

</details>

<!------------------------------------------------------------------------>

<details><summary>Docker Build</summary>

## Build
`image_name` is free as docker image name．<br>
```
cd Dockerfile
sudo docker build -t image_name .
cd ..
```

## Run Container
`$pwd` is mount current dir．<br>
```
sudo docker run -it --shm-size 2g --gpus all -v $(pwd):/workspace image_name 
```

</details>

<!------------------------------------------------------------------------>

<details><summary>Run</summary>
  
`--master_port` is free number <br>

```
cd main
torchrun --master_port 7777 MyoRegenTrack.py
```

</details>

<!------------------------------------------------------------------------>
<!------------------------------------------------------------------------>
<!------------------------------------------------------------------------>

# Run on your data
You can use some args as below.
[file path](https://github.com/RyuAmakaze/MyoRegenTrack/blob/main/main/MyoRegenTrack.py#L242-L251)<br>
[inference param](https://github.com/RyuAmakaze/MyoRegenTrack/blob/main/main/MyoRegenTrack.py#L242-L251)<br>

<details><summary>one image data</summary> 
MyoRegenTrack can run ".png", ".jpg", and ".tif".<br>

```
torchrun --master_port 7777 MyoRegenTrack.py \
--INPUT_PATH path/to/img
```

</details>

<!------------------------------------------------------------------------>

<details><summary>multi-image data</summary> 

Please assign a folder including images.<be>

```
torchrun --master_port 7777 MyoRegenTrack.py \
--INPUT_FOLDER path/to/folder
```

</details>
