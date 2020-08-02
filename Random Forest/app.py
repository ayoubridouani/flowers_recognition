import cv2
import flask
import werkzeug
from skimage import feature
import pickle
import mahotas
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from skimage.color import rgb2hsv
bins= 8

def color_Moments(img):
    img=rgb2hsv(img)
    R=img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]
    colorFeatures=[np.mean(R[:]),np.std(R[:]),np.mean(G[:]),np.std(G[:]),np.mean(B[:]),np.std(B[:])]
    colorFeatures=colorFeatures/np.mean(colorFeatures)
    return colorFeatures
	
def fd_HSV(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([np.uint8(image)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().tolist()

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
	
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
	
def glcm_props(patch):
    lf = []
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']
    patch=np.array(rgb2gray(patch), int)
    # left nearest neighbor
    glcm = feature.greycomatrix(patch, [1], [0], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( feature.greycoprops(glcm, f)[0,0] )
    # upper nearest neighbor
    glcm = feature.greycomatrix(patch, [1], [np.pi/2], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( feature.greycoprops(glcm, f)[0,0] )
    return np.asarray(lf)

app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    fixed_size = tuple((500, 500))
	    
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
	
    image=cv2.imread("androidFlask.jpg")
    try :
	    image = cv2.resize(filename, fixed_size)
    except:
	    print("except")
    ####################################
    # Global Feature extraction
	####################################
    color_moment  = color_Moments(image)
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    histogram     = fd_HSV(image)
    glcm          = glcm_props(image)
    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([color_moment,fv_hu_moments,fv_haralick,histogram,glcm])
    global_feature = np.array(global_feature).reshape(-1,1)
	# scale features in the range (0-1)
    #scaler            = MinMaxScaler(feature_range=(0, 1))
    #rescaled_features = scaler.fit_transform([global_feature])
    #print(rescaled_features.shape)
    #labels=['Calendula', 'Coquelicot', "Feuille d'olivier", 'Feuilles de figuier', 'Glebionis coronaria', 'Jasmin', 'La verveine', 'Menthe', 'Ortie', 'Persil', 'Romarin', 'Sauge', 'Thym', 'lavande']
    with open("model.pkl","rb")as file :
	    model=pickle.load(file)
	    res=model.predict(global_feature.reshape(1,-1))
    return res[0]

app.run(host="0.0.0.0", port=5000, debug=True)