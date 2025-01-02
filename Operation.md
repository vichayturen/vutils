# 操作方法

## 打包tag上传到github
```shell
export new_tag=`cat VERSION`
git tag $new_tag
git push origin $new_tag
```

## 上传到pypi
```shell
python3 setup.py sdist bdist_wheel
twine upload dist/*
```
