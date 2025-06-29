This code parses, plots, and in accordance with the symmetry axes (or PCA axes if none are found) found, scales each shape individually, allowing to adjust the pad size without distorting the overall size.

## -- HOW  TO USE --  
-Pick your Gerber file, either through the browse option or via pasting its directory to the box manually.  
Note: This code is meant to handle singular layers, so export the gerber plot file of the layer in question by itself, ZIP archives work, but you're going to have to direct the code not to the archive, but the layer itself. like "Gerber_TopPasteMaskLayer.GTP".  
-Run Visualization -- this will display your gerber as detected, and if the gerber is plotted via paths, it should display the symmetry axes. It won't displa themin the case of apertures, but they are present all the same.  
-Adjust scaling as you wish. Two scaling factors are given which govern two orthogonal axes. It might not be a good idea to increase or decrease these values too much.  
-(If scaled) run visualization again to see its effects, both non scaled and scaled versions will be displayed.  
-Export either as Dxf or Pdf.

## -- EXPORT AS DXF --  
-Not much to mention, merely pressing the button will yield the results you need.  

## -- EXPORT AS PDF -- 
-When you click, you will be face-to-face with an A4 plane. You will have three parameters to work with, Translate X, Translate Y, and fiducial marks percent.  
-Translate x and y are related to the position of the shapes on the page, so adjust as you wish, the view will update as you change the value.  
-Fiducial marks percent is about the distance the "circles" have from the center of the bounding box, such a feature is optional, and are there purely for calibration. The default value of 80% might not be ideal, but I found 130% to be pretty good.  
-If the marks are deemed unnecessary, setting them outside the bounding box should remove them, as they are rendered as white.  
-After the parameters are set, "Export Final PDF" option will yield the pdf export.  

### Dependancies
matplotlib: For plotting and visualization.  
numpy  
scikit-learn: For PCA (Principal Component Analysis).  
shapely: For geometric operations (LineString, box, Point, Polygon, unary_union, etc.).  
scipy: For scientific computing, specifically interp1d (interpolation).  
ezdxf: For working with DXF CAD files.  
reportlab: For generating PDF documents.  
