# Initialize renv for this project
# Run this script once to set up the renv environment

# Install renv if not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Initialize renv in this directory
renv::init(bare = TRUE)

# Install all packages from the lockfile
renv::restore()

# Alternatively, if you want to install packages fresh and create a new lockfile:
# renv::install(c(
#   "tidyverse",
#   "brms",
#   "ggh4x",
#   "patchwork",
#   "ggforce",
#   "knitr",
#   "kableExtra",
#   "quarto",
#   "scales"
# ))
# renv::snapshot()

cat("\n")
cat("========================================\n")
cat("renv environment initialized!\n")
cat("========================================\n")
cat("\n")
cat("The following packages are now available:\n")
cat("- tidyverse (dplyr, ggplot2, readr, tidyr, etc.)\n")
cat("- brms (Bayesian regression models)\n")
cat("- ggh4x (ggplot2 extensions for nested facets)\n")
cat("- patchwork (combining plots)\n")
cat("- ggforce (additional ggplot2 geoms)\n")
cat("- knitr & kableExtra (tables)\n")
cat("- quarto (for rendering analysis.qmd)\n")
cat("\n")
cat("To activate renv in future R sessions, the .Rprofile\n
")
cat("will automatically source renv/activate.R\n")
cat("\n")
