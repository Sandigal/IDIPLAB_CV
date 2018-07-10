.. _faq:

Frequently Asked Questions
==========================



个人备忘录
=====================================================================


Python
---------------------------------------------------------------
.. code-block:: python3

   # python起文件服务器
   python3.4 -m http.server
   python -m SimpleHTTPServer    # python2

   python -u script.py   # 刷新缓冲，执行脚本重定向结果到文件时候比较有用

   # logging
   FATAL(50) > ERROR(40) > WARNING(30) > INFO(20) > DEBUG(10)

   # 使用virtualenv制定python版本
   virtualenv -p /usr/bin/python2.7 ENV2.7

   # pyenv 安装多个版本的 python : https://github.com/pyenv/pyenv
   # pyenv-virtualenv https://github.com/pyenv/pyenv-virtualenv

   # 格式化 json
   cat some.json | python -m json.tool

   # brew install youtube-dl
   # https://askubuntu.com/questions/486297/how-to-select-video-quality-from-youtube-dl
   # http://www.cnblogs.com/faunjoe88/p/7810427.html
   youtube-dl -F "http://www.youtube.com/watch?v=P9pzm5b6FFY"
   youtube-dl -f 22 "http://www.youtube.com/watch?v=P9pzm5b6FFY"
   youtube-dl -f bestvideo+bestaudio "http://www.youtube.com/watch?v=P9pzm5b6FFY"

Tmux
-------------------------------------------------------------

.. code-block:: shell

   tmux rename -t oriname newname
   tmux att -t name -d               # -d 不同窗口全屏
   # 如果手贱在本机tmux里又ssh到服务器又进入服务器的tmux怎么办
   c-b c-b d

   # Vim style pane selection
   bind -n C-h select-pane -L
   bind -n C-j select-pane -D
   bind -n C-k select-pane -U
   bind -n C-l select-pane -R

   # https://stackoverflow.com/questions/22138211/how-do-i-disconnect-all-other-users-in-tmux
   tmux a -dt <session-name>


