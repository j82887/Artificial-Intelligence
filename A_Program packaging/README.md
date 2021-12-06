## 01. Pyinstaller
#### PyInstaller是將Python應用程序及其所有依賴項捆綁到一個包中。
#### 使用者無需安裝Python或任何套件即可運行打包後的應用程序
* 安裝則在命令提示字元中輸入 `pip install pyinstaller`
* `pyinstaller -h` 查看功能與指令
* 常用指令：
  * -F 打包成一個exe執行檔案
  * –icon=path 給予圖標路徑，預設為內建pyinstaller icon
  * -w 使用視窗，無控制台
  * -c 使用控制台，無視窗
  * -D 創建一個目錄資料夾，包含exe執行檔案及其他一些依賴性文件
  * 加入--hiddenimport手動匯入套件包 安裝完後找不到套件用，如：`pyinstaller -F --hiddenimport [套件名稱] [要打包的.py檔]`
