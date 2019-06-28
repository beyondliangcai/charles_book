gitbook build
git checkout master
git add .
git commit -m $1
git push -u origin master
git checkout gh-pages
gitbook build
cp -r _book/* .
git add .
git commit -m $1
git push -u origin gh-pages
git checkout master