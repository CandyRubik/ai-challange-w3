# Деплой погодного дашборда

Workflow `.github/workflows/ci.yml` запускает проверки и затем синхронизирует
приложение на VDS при push в `main` или `trunk`. Pull request только запускает
проверки. Секреты не копируются из checkout: `.env`, `.venv` и `data/` остаются
на сервере. После синхронизации обновляются зависимости и перезапускаются оба
systemd-сервиса.

Это выкладка только на личный VDS владельца репозитория, а не публичный
хостинг для всех пользователей. Actions variables/secrets для VDS уже настроены
в репозитории; теперь успешный push в `main` или `trunk` запускает job Deploy
после прохождения CI. Блок ниже нужен для проверки и повторной настройки доступа.

## Однократная настройка GitHub Actions

Открой [Actions variables](https://github.com/CandyRubik/ai-challange-w3/settings/variables/actions)
и добавь repository variables:

- `VDS_HOST` = `91.188.214.71`;
- `VDS_USER` = `user`;
- `VDS_KNOWN_HOSTS` = `91.188.214.71 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIN5YRs/5++i5tP/yMpiqdOH2LuRt5RBI3DnYXsQxmMQF`.

В [Actions secrets](https://github.com/CandyRubik/ai-challange-w3/settings/secrets/actions)
добавь repository secret `VDS_DEPLOY_KEY` со всем содержимым закрытого файла
`~/.ssh/ai-challenge-vds-deploy`. На macOS можно скопировать его прямо в буфер
обмена, не печатая в терминал:

```bash
pbcopy < ~/.ssh/ai-challenge-vds-deploy
```

Публичная часть этого deploy-ключа уже установлена в `authorized_keys` VDS.
Личный ключ `h3llo-cloud-ssh-key` в GitHub Actions не используется.

Проверенный публичный ключ VDS закреплён в переменной `VDS_KNOWN_HOSTS`.
При штатном обновлении кода Actions синхронизирует файлы в
`~/ai-challange-w3/`, устанавливает зависимости и перезапускает сервисы.

## Кто может открыть VDS

Приложение на VDS слушает `127.0.0.1:8000`, поэтому оно не опубликовано в
интернете. Приложение доступно только владельцу через SSH-туннель.

Пользователям, которым нужен собственный экземпляр, не нужны SSH-доступ,
VDS-секреты или доступ к базе владельца: они клонируют репозиторий и запускают
web и worker на своей машине по шагам в разделе «Локальный запуск» в README.
Их SQLite и чаты будут отдельными.
