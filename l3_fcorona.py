import numpy as np
from scipy import ndimage
from scipy import interpolate
from astropy.io import fits
import matplotlib.pyplot as plt
import aspiics_misc      as am
import argparse
import os
import sys


def f_corona(xx,yy,**kwargs):
    """Gives the current model of F-corona in MSB interpolated to xx and yy [R_Sun] 2D arrays
    """
    #pixscale=2.8125 ; x_IO=1023.5 ; y_IO=1023.5 ; RSun=16.0*60.0
    #xx = np.outer(np.ones(2048),np.linspace(0,2047,num=2048)-x_IO) * pixscale / RSun
    #yy = np.outer(np.linspace(0,2047,num=2048)-y_IO,np.ones(2048)) * pixscale / RSun

    model=kwargs.get('model','standard')
    if model=='simple_sh' or model=='Allen':
        ## Simple polar-symmetrical model of F-corona from Allen 1977 used by sshestov in his initial
        ##    simulated data IDL software (b_corona.pro),  units - [1e-10 MSB]
        ##               *      *    *                    - these three are fake, to simplify interpolation inside 1.1R_Sun
        r_C = np.array([0.01,  0.5, 0.90, 1.01, 1.03, 1.06,  1.10, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50, 3.00, 4.00, 5.00, 10.0])
        B_F1= np.array([3.27, 3.26, 3.27, 3.22, 3.16, 3.06,  3.00, 2.80, 2.46, 2.24, 2.06, 1.93, 1.81, 1.65, 1.43, 1.10, 0.83, 0.23])-10.0
        #  from my IDL code       R_Sun =[1.01, 1.03, 1.06,  1.10, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50, 3.00, 4.00, 5.00, 10.0] ; Allen
        #                         B_F_A =[3.22, 3.16, 3.06,  3.00, 2.80, 2.46, 2.24, 2.06, 1.93, 1.81, 1.65, 1.43, 1.10, 0.83, 0.23]
        B_F2= B_F1.copy() #np.array([3.25, 3.24, 3.23, 3.22, 3.16, 3.06,  3.00, 2.80, 2.46, 2.24, 2.06, 1.93, 1.81, 1.65, 1.43, 1.10, 0.83, 0.23])-10.0
        origin='Allen 1977'
        
    else:      # implying standard model=='standard':          
        ## Brightness of the F-corona, Koutchmy (2000); units - [1e-10 MSB]
        ##                *     *      *    *      *     - these five are fake, to simplify interpolation inside 1.1R_Sun
        r_C  = np.array([0.1,  0.5,  0.95, 1.03, 1.06,  1.10, 1.20, 1.40, 1.60, 2.00, 2.50, 3.00, 4.00, 5.00,10.0])
        B_F1 = np.array([3.25, 3.24, 3.23, 3.21,  3.2,  3.10, 2.90, 2.50, 2.25, 1.91, 1.66, 1.48, 1.23, 1.00, 0.31])-10.0
        B_F2 = np.array([3.25, 3.23, 3.23, 3.21,  3.2,  3.10, 2.90, 2.50, 2.25, 1.82, 1.56, 1.33, 1.03, 0.80, 0.06])-10.0
        origin='Koutchmy2000'
    
    
    rr = np.sqrt( np.add(np.square(xx),np.square(yy)) )
    phi= np.arctan2(yy,xx)
    c1= np.abs(np.abs(phi)-np.pi/2.)/(np.pi/2.)
    c2= 1.0 - c1

    kind='linear'      #  ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’
    inter1 = interpolate.interp1d(r_C, B_F1, kind=kind, fill_value='extrapolate')   # bounds_error="False", fill_value=3.2)  gives error?
    inter2 = interpolate.interp1d(r_C, B_F2, kind=kind, fill_value='extrapolate')   

    Fcor1 = inter1(rr)
    Fcor2 = inter2(rr)
    Fcor = c1*Fcor1 + c2*Fcor2
    Fcor = np.power(10.0,Fcor)
    Fcor = Fcor.astype(np.float32)

    verbose=kwargs.get('verbose', False)
    if verbose:
        plt.plot(rr[1024,:],Fcor[1024,:],'b',label="Interpolated horiz. F-cor")
        plt.plot(rr[:,1024],Fcor[:,1024],'r',label="Interpolated vert. F-cor")
        plt.plot(r_C,np.power(10.,B_F1),'-o',label="Tabulated horiz. "+origin)
        plt.plot(r_C,np.power(10.,B_F2),'-*',label="Tabulated vert.  "+origin)
        plt.yscale('log')
        plt.ylim(1e-9,1e-6)        
        plt.xlim(0.0,3.0)
        plt.legend()
        plt.show()

    return Fcor, origin, kind
    

#def l3_fcorona(inputfile):
print("%*******************************************************************************")
print("% L3 F-corona removal: processing ")

# Initialize parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("file", help="input Level-3 ASPIICS Wideband fits")
parser.add_argument("--outdir", help="force output dir (default './output/')", default='./output/')
#parser.add_argument("--center", help="re-center/de-rotate the images", default=True, action=argparse.BooleanOptionalAction)
#Parse arguments
args = parser.parse_args()

outputdir = args.outdir
#docenter = args.center

# receive input filenames from the command line. We assume original exposure time was 1<2<3, which is encoded by 01, 1 and 10 in var names
file =args.file  #sys.argv[1]    #file10='tile_map/ASPIICS_synthetic_T30S3000_10.0sec_filterWB.fits'
print("    Input file:")
print("     file: ",file)

data, header = am.read_fits_image_array(file)

pixscale = header['CDELT1']
CRPIX1   = header['CRPIX1']
CRPIX2   = header['CRPIX2']
#CRPIX1   = header['X_IO']-1.0  # !!!! to put back !!!! header['CRPIX1']-1.0            # these are center of the Sun in the image, re-centered during l3_merge
#CRPIX2   = header['Y_IO']-1.0  # header['CRPIX2']-1.0
RSUN_ARC = header['RSUN_ARC'] 

#pixscale=2.8125 ; x_IO=1023.5 ; y_IO=1023.5 ; RSun=16.0*60.0
xx = np.outer(np.ones(2048),np.linspace(0,2047,num=2048)-CRPIX1) * pixscale / RSUN_ARC
yy = np.outer(np.linspace(0,2047,num=2048)-CRPIX2,np.ones(2048)) * pixscale / RSUN_ARC

#Fcor, Fcor_msg, Fcor_kind = f_corona(xx,yy,model='simple_sh')  ### --- Sergei's data were created with Allen model ### ,verbose=True --- with plots
Fcor, Fcor_msg, Fcor_kind = f_corona(xx,yy,model='standard')    ### --- Koutchmy et al 2002  ### ,verbose=True --- with plots
data=data-Fcor

if 'F_COR' in header:
    del header['F_COR']

headerF = header.copy()
header.set('PROD_ID','K-corona',comment='F-cor from '+Fcor_msg+'; kind='+Fcor_kind,after='FILTER')
header.set('HISTORY','F-corona removal l3_fcorona.py')
header.set('HISTORY','F-corona model: '+Fcor_msg+'; interpolation method (kind): '+Fcor_kind)
header.set('PARENT',file)


headerF.set('PROD_ID','F-corona model',comment='F-cor from '+Fcor_msg+'; kind='+Fcor_kind,after='FILTER')
headerF.set('HISTORY','F-corona removal l3_fcorona.py')
headerF.set('HISTORY','F-corona model: '+Fcor_msg+'; interpolation method (kind): '+Fcor_kind)

# writing main file with removed corona
hdu = fits.PrimaryHDU(data,header)
newname = os.path.splitext(os.path.basename(file))[0]+'.Fcor_removed.fits'
file2write = os.path.join(outputdir,newname)
if os.path.isfile(file2write):
   print("% L3_Fcorona. Output file "+file2write+" exists. Removing it")
   os.remove(file2write)
print("% L3_Fcorona. Writing "+file2write)
hdu.writeto(file2write)

## writing F-corona model
#hdu = fits.PrimaryHDU(Fcor,headerF)
#file2write = 'output/'+os.path.splitext(os.path.basename(inputfile))[0]+'.Fcor_model.fits'
#if os.path.isfile(file2write):
#   print("% L3_Fcorona. Output file "+file2write+" exists. Removing it")
#   os.remove(file2write)
#print("% L3_Fcorona. Writing "+file2write)
#hdu.writeto(file2write)
    


#if __name__ == '__main__':
#    inputfile = sys.argv[1]
#    l3_fcorona(inputfile)
 
