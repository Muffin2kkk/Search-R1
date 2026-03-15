# 公用服务器上传 Git 代码流程

这份文档总结了在公用服务器上向 GitHub 推送代码的推荐流程，适用于当前仓库和当前账号。

关键信息：

- GitHub 用户名：`Muffin2kkk`
- 提交邮箱：`1045008115@qq.com`
- 仓库地址：`https://github.com/Muffin2kkk/Search-R1.git`

## 核心经验

1. `git config user.name` 和 `git config user.email` 只影响提交作者信息，不决定 `git push` 时使用的认证身份。
2. 公用服务器上不要长期保存 SSH 私钥，也不要把 PAT 写进配置文件、脚本、远端 URL 或命令历史。
3. Cursor/VS Code/Cursor Server 可能会通过 `GIT_ASKPASS` 自动注入旧凭证，导致出现误导性的 `403` 或 `account suspended` 报错。
4. 在公用服务器上，推荐做法是：每次需要 `push` 时临时新建一个 PAT，用完后立刻删除。

## 推荐流程

### 1. 进入仓库

```bash
cd ~/Search-R1
```

### 2. 如果 GitHub 连通性不稳定，先启用网络加速

```bash
source /etc/network_turbo
```

如果终端输出 `设置成功`，说明加速脚本已生效。

### 3. 检查仓库状态和远端地址

```bash
git status
git remote -v
```

预期远端应类似：

```bash
origin  https://github.com/Muffin2kkk/Search-R1.git (fetch)
origin  https://github.com/Muffin2kkk/Search-R1.git (push)
```

### 4. 只给当前仓库配置提交作者

公用服务器上尽量不要使用 `--global`，只配置当前仓库：

```bash
git config user.name "Muffin2kkk"
git config user.email "1045008115@qq.com"
```

### 5. 提交代码

全部提交：

```bash
git add .
git commit -m "你的提交说明"
```

如果只想提交部分文件：

```bash
git add 某个文件
git commit -m "你的提交说明"
```

### 6. 去 GitHub 网页临时新建一个 PAT

推荐使用 fine-grained PAT，并按最小权限配置：

- Repository access: `Only select repositories`
- 选择仓库：`Search-R1`
- Permissions:
- `Contents: Read and write`

建议只在需要推送时临时创建，用完立刻删除。

### 7. 用干净认证方式推送，绕过 Cursor/VS Code 残留凭证

```bash
env -u GIT_ASKPASS \
    -u SSH_ASKPASS \
    -u VSCODE_GIT_ASKPASS_NODE \
    -u VSCODE_GIT_IPC_AUTH_TOKEN \
    -u VSCODE_GIT_ASKPASS_EXTRA_ARGS \
    -u VSCODE_GIT_IPC_HANDLE \
    -u VSCODE_GIT_ASKPASS_MAIN \
    GIT_TERMINAL_PROMPT=1 \
    git push
```

执行后按提示输入：

```bash
Username for 'https://github.com': Muffin2kkk
Password for 'https://Muffin2kkk@github.com':
```

注意：

- `Username` 填：`Muffin2kkk`
- `Password` 位置填写的是刚创建的 `PAT`
- 这里不要输入 GitHub 网页登录密码

### 8. 成功时的典型输出

```bash
To https://github.com/Muffin2kkk/Search-R1.git
   5789e52..a2cfa62  main -> main
```

### 9. 推送完成后立刻删除 PAT

在 GitHub 网页删除刚才创建的 PAT。下次需要 `push` 时，再重新创建一个新的短期 PAT。

## 以后可直接使用的最小流程

```bash
cd ~/Search-R1
source /etc/network_turbo
git status
git add .
git commit -m "你的提交说明"
env -u GIT_ASKPASS \
    -u SSH_ASKPASS \
    -u VSCODE_GIT_ASKPASS_NODE \
    -u VSCODE_GIT_IPC_AUTH_TOKEN \
    -u VSCODE_GIT_ASKPASS_EXTRA_ARGS \
    -u VSCODE_GIT_IPC_HANDLE \
    -u VSCODE_GIT_ASKPASS_MAIN \
    GIT_TERMINAL_PROMPT=1 \
    git push
```

输入时：

- 用户名：`Muffin2kkk`
- 密码：临时新建的 `PAT`

## 公用服务器安全原则

- 不要把长期使用的 SSH 私钥上传到公用服务器。
- 不要执行 `git config --global credential.helper store`。
- 不要把 PAT 写进 `~/.bashrc`、脚本、远端 URL 或命令历史。
- 不要把 GitHub 登录密码当作 `git push` 的密码。
- 优先使用短期、最小权限、现建现删的 PAT。

## 常见报错判断

### 1. `Author identity unknown`

说明没有配置提交作者，执行：

```bash
git config user.name "Muffin2kkk"
git config user.email "1045008115@qq.com"
```

### 2. `remote: Your account is suspended`

如果你确认自己的 GitHub 账号正常，这通常不是账号本身有问题，而是共享环境或编辑器自动注入了旧凭证。优先改用上面的“干净认证”方式推送。

### 3. `Permission denied`

通常表示：

- 输入的是 GitHub 登录密码，不是 PAT
- 或者 PAT 没有目标仓库的写权限

### 4. `503`

更像网络或代理问题，可先执行：

```bash
source /etc/network_turbo
```

## 一次完整实例

```bash
cd ~/Search-R1
source /etc/network_turbo
git config user.name "Muffin2kkk"
git config user.email "1045008115@qq.com"
git status
git add .
git commit -m "Update inferonly"
env -u GIT_ASKPASS \
    -u SSH_ASKPASS \
    -u VSCODE_GIT_ASKPASS_NODE \
    -u VSCODE_GIT_IPC_AUTH_TOKEN \
    -u VSCODE_GIT_ASKPASS_EXTRA_ARGS \
    -u VSCODE_GIT_IPC_HANDLE \
    -u VSCODE_GIT_ASKPASS_MAIN \
    GIT_TERMINAL_PROMPT=1 \
    git push
```

交互时输入：

```bash
Username for 'https://github.com': Muffin2kkk
Password for 'https://Muffin2kkk@github.com': 这里输入临时创建的 PAT
```
