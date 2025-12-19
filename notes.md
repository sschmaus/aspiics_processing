# ASPIICS processing notes

## L2 calibration

* arbitrary gain value
* linearity correction bug:
  * too large values (why do they exist) will not be corrected
  * these values are not marked as overexposed, resulting in halo artefact
  * saturation detection should use max value from non-linearity math not something else
* flat is not bias corrected
* flat doesn't correct all spots

## Line noise ideas

* line filter fix:
* subtract 2d median to remove continuum
* apply 1d median to remove stars etc..
* average lines (possibly split lr)
* subtract median to remove large remaining features
* separate alternating collumns
