#!/bin/sh

# Credits: http://stackoverflow.com/a/750191

git filter-branch -f --env-filter "
    GIT_AUTHOR_NAME='Pranav Chaturvedi'
    GIT_AUTHOR_EMAIL='pranavchat14@gmail.com'
    GIT_COMMITTER_NAME='Pranav Chaturvedi'
    GIT_COMMITTER_EMAIL='pranavchat14@gmail.com'
  " HEAD