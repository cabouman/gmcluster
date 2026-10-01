# Per-package settings for the dev scripts.  This is the only dev script that
# differs between packages; all the others are identical across repositories.
NAME="gmcluster"
PYTHON_VERSION="3.11"
EXTRAS="test,docs"          # pip extras installed with the editable package

# Optional hooks: define a function to run extra steps, or leave it out to skip.
# before_env_create() { : ; }   # runs before the env is created (e.g. module load)
# extra_clean()       { : ; }   # runs during remove_package (e.g. a cache)
