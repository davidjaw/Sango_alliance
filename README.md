# 三國志戰略版 同盟戰功擷取工具

## Demo video

https://www.youtube.com/watch?v=3twTnTO2wRY

## 安裝
* 請先下載 [Visual Studio 函式庫](https://learn.microsoft.com/zh-tw/cpp/windows/latest-supported-vc-redist?view=msvc-170)
* 下載 [python 3.10](https://www.python.org/downloads/)
* 下載 [git](https://git-scm.com/downloads)

1. `clone` this repo!
```ps1
git clone https://github.com/davidjaw/Sango_alliance.git
```

2. 在當前資料夾開啟你的 `powershell` 創建虛擬環境
```ps1
python -m venv venv
```

3. 啟動虛擬環境
```ps1
.\venv\Scripts\activate
```

4. 根據你的 CUDA 版本安裝 pytorch
```ps1
# CUDA 11.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU (沒有GPU的人用這個)
pip3 install torch torchvision torchaudio
```

5. 安裝其他套件
```ps1
pip3 install -r requirements.txt
```

