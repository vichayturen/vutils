# 操作方法

## 打包tag上传到github
```shell
export new_tag=v0.0.1
git tag $new_tag
git push origin $new_tag
```

## 上传到pypi
```shell
python3 setup.py sdist bdist_wheel
twine upload dist/*
```
