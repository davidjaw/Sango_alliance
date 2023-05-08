# 三國志戰略版 同盟戰功擷取工具

贊助我一杯咖啡:

<a href="https://www.buymeacoffee.com/jdway"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=jdway&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" /></a>

* 目前僅適用於 三國志戰略版 的電腦版，僅在 Windows 10 測試過，因為我電腦算好，執行時間才會非常快
* 有任何問題可以到 [Github Issue](https://github.com/davidjaw/Sango_alliance/issues), 巴哈文章 或者 Discord 找 `Midori Neko#6514`

## Demo video

https://www.youtube.com/watch?v=3twTnTO2wRY
[![Demo video](https://raw.githubusercontent.com/davidjaw/Sango_alliance/main/img/video_func.png)](https://www.youtube.com/watch?v=3twTnTO2wRY)

* 可以透過按跳出的 `tk` 視窗進行辨識，或者透過按下空白鍵進行辨識
  * 程式開始執行後要先等待 `tk` 視窗跳出來，他有時候不會跳到最前面
* 中文辨識方面目前是使用 [CnOCR](https://github.com/breezedeus/CnOCR) 套件，辨識度頗低，但是還是能勉強辨識到ID
  * 可以從座標去判斷到底是誰
  * 截圖的同時也會把原圖存到 `tmp/sc` 和 `tmp/sc-full`, 都是有跡可循
* 開始執行後記得不要再移動遊戲視窗、也不要對視窗縮放
  * 如果不小心動到，可以把程式關了再重新執行

## 安裝
有很多套件要預先下載 & 安裝
* 下載 [python 3.10](https://www.python.org/downloads/)
* 下載 [git](https://git-scm.com/downloads)


### 參考用安裝影片 (請先下載並安裝 VS 函式庫、python 和 git):

https://www.youtube.com/watch?v=pCnBXA1c80c
[![install video](https://raw.githubusercontent.com/davidjaw/Sango_alliance/main/img/video_install.png)](https://www.youtube.com/watch?v=pCnBXA1c80c)

### 安裝步驟

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

* 可以用 `nvidia-smi` 指令來確認你的 CUDA 版本, 參考安裝影片, 如果版本高於 11.8 (例如我是 12.x), 安裝 11.8 版本應該就可以 (~~至少我自己沒有問題~~)

```ps1
# CUDA 11.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU (沒有GPU的人用這個)
pip3 install torch torchvision torchaudio
```

5. 先安裝該死的 `Polygon3`
* 這個 library compile 有一些 C++ 套件相關的坑，這是我 compile 過的檔案，目前測試 win10, win11 應該是能直接用這個檔案安裝，如果有安全疑慮也可以略過這步驟自己用 `pip` compile
  * 特別感謝 `宇佐美暁` 幫忙花一堆時間踩坑測試
```ps1
pip install .\whl\Polygon3-3.0.9.1-cp310-cp310-win_amd64.whl
```

6. 安裝其他套件
```ps1
pip3 install -r requirements.txt
```

## Troubleshooting
### 1. 執行 `./venv/Scripts/activate` 時報錯
  * 如果是說找不到 `activate`，將命令改成 `./venv/Scripts/Activate.ps1` 或 `./venv/Scripts/activate.bat`
  * 如果是說`venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system. For more information, see about_Execution_Policies`, 先嘗試解法 1，沒效再試試看解法 2
    * 參考 [解法1](https://dev.to/aka_anoop/enabling-virtualenv-in-windows-powershell-ka3)，在 powershell 輸入 `Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser`
    * 參考 [解法2](https://stackoverflow.com/questions/18713086/virtualenv-wont-activate-on-windows) 在 powershell 輸入 `Set-ExecutionPolicy Unrestricted -Scope Process`



