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
  * --hiddenimport 手動加入匯入套件包(打包完後找不到套件時用)，如：`pyinstaller -F --hiddenimport [套件名稱] [要打包的.py檔]`
  * --exclude-module 手動移除套件包，如`pyinstaller --clean -F test.py --exclude-module [套件名稱]`
  * 從上個打包產生的spec檔案，繼續打包： pyinstaller --clean -F test.spec

* **備註** 
* 容易打包後的檔案太大，原因為打包許多不必要的套件。
  * 盡量以`from xxx import xxx`方式打包，相較於`import xxx`來得好
  * 建立一新的環境，只安裝需要使用的套件
  * --exclude-module 手動移除套件包
