# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:29:00 2015

@author: Vijay Anand
"""

import pygtk
pygtk.require("2.0")
import gtk
import copy
import gtk.glade
import cv2
import numpy as np
import os
import pywt
from scipy import spatial
import operator
import os.path
import pickle
from matplotlib import pyplot as plt

###############################################################################

def msgwindow(message,flag):
    
    if flag==0:
    
        parent = None
        md = gtk.MessageDialog(parent,gtk.DIALOG_DESTROY_WITH_PARENT,
                               gtk.MESSAGE_ERROR, gtk.BUTTONS_CLOSE, message)
        md.run()
        md.destroy()
    
    if flag==1:
        parent = None
        md = gtk.MessageDialog(parent,gtk.DIALOG_DESTROY_WITH_PARENT,
                               gtk.MESSAGE_INFO, gtk.BUTTONS_CLOSE, message)
        md.run()
        md.destroy()

###############################################################################
def check_uniqueimgs(superdb,img_sig):
    
    
    flag=[]
    
    for key,value in img_sig.iteritems():
        
        for k1, v1 in superdb.iteritems():
   
            for k2,v2 in v1['sig'].iteritems():
                
                if np.array_equal(value,v2):
                    msg='Image '+key+ ' already exists in database. Please remove '+key+' from the folder and try again !'
                    msgwindow(msg,0)
                    flag=[key]
                    
    return flag

###############################################################################

def decode_img(simimg):
    
    cA=simimg[0]
    cD=simimg[1]
    
    inv=pywt.idwt(cA, cD, 'db1')
    
    img=np.reshape(inv,(384,512,3))
    img=img.astype('uint8')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img
    
###############################################################################
def create_sig(imgs_path):
    imgs_list=os.listdir(imgs_path)
    
    image_db={}
    
    for imgs in imgs_list:
        
        i=cv2.imread(imgs_path+'\\'+imgs)
        #i=cv2.resize(i,(384, 512), interpolation = cv2.INTER_CUBIC)
        i=i.flatten()
        
        image_db[imgs]=i[:2000]
        
    return image_db
    
###############################################################################
    
def clean_duplicates(loc):
    
    imlist=os.listdir(loc)
    
    imdict={}
    
    for im in imlist:
        
        i=cv2.imread(loc+'\\'+im)
        imdict[im]=i
        
    duplicates=[]
        
    ##check duplicates
        
    for k,v in imdict.iteritems():
        
        for key,value in imdict.iteritems():
            
            if key !=k:
                if np.array_equal(v,value):
                    
                    duplicates.append((key,k))
    
    for pair in duplicates:
        
        if os.path.isfile(loc+'\\'+pair[0]) and os.path.isfile(loc+'\\'+pair[1]):
            os.remove(loc+'\\'+pair[1])
            msgwindow('Image duplicate exists. Removed '+pair[1]+' from the folder !',0)

    
###############################################################################
    
def create_image_db(imgs_path):
    
    
    imgs_list=os.listdir(imgs_path)
    
    image_db={}
    
    for imgs in imgs_list:
        
        print 'Processing ',imgs
        i=cv2.imread(imgs_path+'\\'+imgs)
        i=i.flatten()
        cA, cD = pywt.dwt(i, 'db1')
        image_db[imgs]=[cA,cD]
        
    return image_db
    
###############################################################################
    
def extract_binkeys(image_db):
    
    bin_keys={}
    for img_name,wavelet_coeff in image_db.iteritems():
        cD=wavelet_coeff[1]
        key=np.zeros(cD.shape[0],)
        

        for x in range(0,cD.shape[0]):

            if cD[x]>0:
                key[x]=1
                
        bin_keys[img_name]=key
        
   
    return bin_keys
    
###############################################################################
    
def plot_enc_img(encimages):
    
    cnt=0
    splot=[221,222,223,224]    
    fig = plt.gcf()
    fig.canvas.set_window_title('Sample encrypted Images in database')
    
    for key,value in encimages.iteritems():
        
        i=value[1]
           
        x=i[:159600]
      
        x=np.reshape(x,(200,266,3))
        x= cv2.resize(x,(512,384), interpolation = cv2.INTER_CUBIC)
        plt.subplot(splot[cnt]),plt.imshow(x),plt.title('Encrypted image of '+key)
        plt.xticks([]), plt.yticks([])

    
    
        if cnt==3:
            break
    
        cnt+=1
        
    plt.show()
        
###############################################################################
def create_enc_fpdb(fp_path):
    
    fp_db={}
    img=fp_path.split('\\')[-1]
    
    g=cv2.imread(fp_path)
    g=cv2.cvtColor(g,cv2.COLOR_BGR2GRAY)
    kp = sift.detect(g,None)
    #kps=np.zeros(294912)
    kps=np.zeros(500)
    cnt=0
    for k in kp:
        x=int(k.pt[0])
        kps[cnt]=x
        cnt+=1
        y=int(k.pt[1])
        kps[cnt]=y
        cnt+=1

    fp_db[img]=kps
    
    print 'SIFT keypoints of ',img,' : ',kps[:200]
    
   
    return fp_db

###############################################################################

def draw_kp_sift(g):
    
    
    kp = sift.detect(g,None)
    img=cv2.drawKeypoints(g,kp)
    fig = plt.gcf()
    fig.canvas.set_window_title('SIFT keypoints')
    plt.title('SIFT keypoints of fingerprint image')    
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    
    plt.show()
    
    
###############################################################################
    
def cds(testimg,fp_img,superdb):
    
    match_flag=0
    ai_flag=0
    for key,value in superdb.iteritems():
    ## Authenticate
        orig=testimg
        fp_db=value['fp_db']
        
    ## extract keypoints of fp
        kp = sift.detect(fp_img,None)
        kps=np.zeros(500)
        cnt=0
        for k in kp:
            x=int(k.pt[0])
            kps[cnt]=x
            cnt+=1
            y=int(k.pt[1])
            kps[cnt]=y
            cnt+=1
            
        print 'SIFT keypoints of ',fp_img,' : ',kps[:200]
            
    ## Authenticate
        
        kps=np.sort(kps)    
        match_flag=0
    
        for filename,fp in fp_db.iteritems():            
            fp=np.sort(fp)
                
            if np.array_equal(fp,kps):            
                match_flag=1
            
   
    
        if match_flag==0:
        
            pass
        
        else:
                
            ## Find closest matching image in db
            testimg=np.ndarray.flatten(testimg)
            cA, cD = pywt.dwt(testimg, 'db1')
            
            key=np.zeros(cD.shape[0],)
        
            for x in range(0,cD.shape[0]):
        
                    if cD[x]>0:
                        key[x]=1
                        
            print 'Binary key of test image : ',key
            
            corr_dict={}
            
            bin_keys=value['Binary keys']
                
            for im,bin_key in bin_keys.iteritems():
                
                corr_dict[im]=spatial.distance.hamming(key, bin_key)
                #corr_dict[im]=sim_score(key,bin_key)
            
                
            ## find min of dict
            
            closest_match=min(corr_dict.iteritems(), key=operator.itemgetter(1))[0]
            
            #closest_match=max(corr_dict.iteritems(), key=operator.itemgetter(1))[0]
                
            ## plot the images
            
            imgdb=value['Images']
            simimg=imgdb[closest_match]
            
            simage=decode_img(simimg)
            orig=cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            
            score=corr_dict[closest_match]
            
            if score <0.4:
            
                simscore='Similarity score : '+str(corr_dict[closest_match])+' : Similar image exists in database'
                fig = plt.gcf()
                fig.canvas.set_window_title(simscore)
                plt.subplot(122),plt.imshow(orig),plt.title('Test Image')
                plt.xticks([]), plt.yticks([])
                plt.subplot(121),plt.imshow(simage),plt.title('Closest match in database')
                plt.xticks([]), plt.yticks([])
                plt.show()
                ai_flag=1
                
            break
            
    
    if match_flag==0:
        msgwindow('Authentication failed !',0)
        return
        
    elif ai_flag==0:
        msgwindow('Test image does not exist in database !',1)
        

###############################################################################

class cbcd_ui:

    def __init__(self):
        self.gladefile = "ui.glade" 
        self.glade = gtk.Builder()
        self.glade.add_from_file(self.gladefile)
        self.glade.connect_signals(self)
        self.window=self.glade.get_object("home")
        self.window.show_all()

        #if (self.window):
        self.window.connect("destroy", gtk.main_quit)
        
        ## dictionary of functions to be executed
        dic={'on_button1_clicked':self.on_button1_clicked,
             'on_button2_clicked':self.on_button2_clicked,
             'on_button3_clicked':self.on_button3_clicked,
             'on_button4_clicked':self.on_button4_clicked,
             'on_filechooserbutton1_file_set':self.on_filechooserbutton1_file_set,
             'on_filechooserbutton2_file_set':self.on_filechooserbutton2_file_set,
             'on_filechooserbutton3_file_set':self.on_filechooserbutton3_file_set,
             'on_filechooserbutton4_file_set':self.on_filechooserbutton4_file_set}
             
        self.glade.connect_signals(dic)
        
    #### Variables
    
        
    ucid_loc=''
    fingerprint_loc=''
    
    testimage=''
    authenticationimage=''
    
    db={}
    superdb={}
    
    ## check if db.dat exists
    
    try:
        
        if os.path.isfile('db.dat') :            
            pass

                
    except:
        superdb={}
    
    
    ###############      Create database
        
    def on_filechooserbutton1_file_set(self,widget):  ## UCID images folder        
        self.ucid_loc=widget.get_current_folder()
        
    def on_filechooserbutton2_file_set(self,widget):  ## Fingerprint images File        
        self.fingerprint_loc=widget.get_filename()
        
        try:
            fi=cv2.imread(self.fingerprint_loc)
            fi=cv2.cvtColor(fi,cv2.COLOR_BGR2GRAY)
            draw_kp_sift(fi)
        
        except:
            pass
        
    def on_button1_clicked(self, widget):  ## create db button
        
        ## check if 4 users exists in superdb
        
        if os.path.isfile('db.dat') :            
            
            with open('db.dat', 'rb') as handle:
                self.superdb = pickle.load(handle)
                
            if len(self.superdb)==4:
                       
                msgwindow('Limit on maximum number of users reached ! To create a new database,please delete the db.dat file.',1)                
                return
                
            else:
                
                dbkey='user'+str(len(self.superdb)+1)
                
                ## get test image and db image location
            if self.ucid_loc==None or self.fingerprint_loc==None:                
                msgwindow('Please set a valid UCID/Fingerprint images folder !',0)
            
            elif len(self.ucid_loc) ==0 or len(self.fingerprint_loc)==0:                
                msgwindow('Please set a valid UCID/Fingerprint images folder !',0)
            
            elif not os.path.isdir(self.ucid_loc):                
                msgwindow('Please set a valid UCID/Fingerprint images folder !',0)
                
            elif  not os.path.isfile(self.fingerprint_loc):
                msgwindow('Please set a valid UCID/Fingerprint images folder !',0)
                                
            else:                ## ready to create database
                
                img_sig=create_sig(self.ucid_loc)                
                flag=check_uniqueimgs(self.superdb,img_sig)
                
                if flag==[]:
                    
                    clean_duplicates(self.ucid_loc)
                    image_db=create_image_db(self.ucid_loc)
                    
                    bin_keys=extract_binkeys(image_db)
                    fp_db=create_enc_fpdb(self.fingerprint_loc)
                
        
                    self.db={}
                    self.db['Images']=image_db
                    self.db['Binary keys']=bin_keys
                    self.db['fp_db']=fp_db
                    self.db['sig']=img_sig
                
                
                    self.superdb[dbkey]=self.db
            
                    with open('db.dat', 'wb') as handle:
                        pickle.dump(self.superdb, handle)
                    
                    msgwindow('Successfully created encrypted database and saved to disk!',1)
                    plot_enc_img(self.db['Images'])
        
        ## create db from scratch if there are no users in database
                
                
            
        else:
            
            ## get test image and db image location
            if self.ucid_loc==None or self.fingerprint_loc==None:                
                msgwindow('Please set a valid UCID/Fingerprint images folder !',0)
            
            elif len(self.ucid_loc) ==0 or len(self.fingerprint_loc)==0:                
                msgwindow('Please set a valid UCID/Fingerprint images folder !',0)
            
            elif not os.path.isdir(self.ucid_loc):                
                msgwindow('Please set a valid UCID/Fingerprint images folder !',0)
                
            elif  not os.path.isfile(self.fingerprint_loc):
                msgwindow('Please set a valid UCID/Fingerprint images folder !',0)
                                
            else:                ## ready to create database
                
                clean_duplicates(self.ucid_loc)
                image_db=create_image_db(self.ucid_loc)
                bin_keys=extract_binkeys(image_db)
                fp_db=create_enc_fpdb(self.fingerprint_loc)
        
                self.db={}
                self.db['Images']=image_db
                self.db['Binary keys']=bin_keys
                self.db['fp_db']=fp_db
                
                img_sig=create_sig(self.ucid_loc)   
                self.db['sig']=img_sig
                
                self.superdb['user0']=self.db
            
                with open('db.dat', 'wb') as handle:
                    pickle.dump(self.superdb, handle)
                
            
                msgwindow('Successfully created encrypted database and saved to disk!',1)                
                plot_enc_img(self.db['Images'])
                
    
    #####################################################            
    ############## Validate test image ##################
    #####################################################
                
    def on_filechooserbutton3_file_set(self,widget):  ## Test images file
        self.testimage=widget.get_filename()
        
        
    def on_filechooserbutton4_file_set(self,widget):  ## Authentication Fingerprint image
        self.authenticationimage=widget.get_filename()
        
        try:
            
            fi=cv2.imread(self.fingerprint_loc)
            fi=cv2.cvtColor(fi,cv2.COLOR_BGR2GRAY)
            draw_kp_sift(fi)
        
        except:
            pass
        
        
    def on_button2_clicked(self, widget):  ## Validate
    
    
        if self.testimage==None or self.authenticationimage==None:                
            msgwindow('Please set a valid test/authentication image !',0)
            
        elif len(self.testimage) ==0 or len(self.authenticationimage)==0:                
            msgwindow('Please set a valid test/authentication image !',0)
            
        elif not os.path.isfile(self.testimage):                
            msgwindow('Please set a valid test/authentication image !',0)
                
        elif not os.path.isfile(self.authenticationimage):
            msgwindow('Please set a valid test/authentication image !',0)
                                
        else:                ## ready to create database
        
            if self.superdb=={}:
                msgwindow('Please create the encrypted database first !',0)
            else:
                
                testimg=cv2.imread(self.testimage)
                  
                fp_img=cv2.imread(self.authenticationimage)
                fp_img=cv2.cvtColor(fp_img,cv2.COLOR_BGR2GRAY)
                    
    ## check if fp is in db and find closest match
    
                cds(testimg,fp_img,self.superdb)
            
            
    ## Close button
    
    def on_button3_clicked(self, widget):
       
        gtk.main_quit()
        
    ## Button to delete the db.dat
        
    def on_button4_clicked(self, widget):
        
        try:
            os.remove('db.dat')
            self.superdb={}
            self.db={}
            self.testimage==''
            self.authenticationimage==''
            self.ucid_loc==''
            self.fingerprint_loc==''
            
            
            msgwindow('Encrypted image database deleted successfully !',1)
        except:
            msgwindow('Unable to delete the encrypted image database(s) !',0)
        
###############################################################################       

if __name__ == "__main__":

    global sift
    
    sift=cv2.SIFT(100)
    
    try:
        os.remove('db.dat')
    except:
        pass

    try:
        a = cbcd_ui()
        gtk.main()
        
    except KeyboardInterrupt:
        pass