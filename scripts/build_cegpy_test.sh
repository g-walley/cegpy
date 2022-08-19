# Move old distribution to the old folder.
mkdir --parents .dist_old/; mv dist/* $_
# build the package
python -m build
# Upload new package
python -m twine upload --repository testpypi dist/*
