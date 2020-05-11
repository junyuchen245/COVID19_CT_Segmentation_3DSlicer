import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import math
import keras
from keras.models import Model, load_model
from keras import backend as K
from scipy import ndimage
import cv2
from scipy import ndimage
from keras.utils import to_categorical

#
# ConvNetCovid
#

class ConvNetCovid(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ConvNetCovid" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# ConvNetCovidWidget
#

class ConvNetCovidWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  ###################### Load ConvNet Keras Model ###################
  cwd = os.getcwd()
  print(cwd)
  segmentation_model = load_model('C:/Users/ASUS/Desktop/COVID19_CT_Segmentation_3DSlicer-master/COVID19_CT_SEGMENTATION/ConvNetCovid/COVIDModel_RFCM_FCM_0.1_1.h5')
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    inputCollapsibleButton = ctk.ctkCollapsibleButton()
    inputCollapsibleButton.text = "Inputs"
    self.layout.addWidget(inputCollapsibleButton)

    outputsCollapsibleButton = ctk.ctkCollapsibleButton()
    outputsCollapsibleButton.text = "Outputs"
    self.layout.addWidget(outputsCollapsibleButton)

    # Layout within the dummy collapsible button
    inputsFormLayout = qt.QFormLayout(inputCollapsibleButton)
    outputsFormLayout = qt.QFormLayout(outputsCollapsibleButton)

    #
    # input CT volume selector
    #
    self.inputSelector_ct = slicer.qMRMLNodeComboBox()
    self.inputSelector_ct.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector_ct.selectNodeUponCreation = True
    self.inputSelector_ct.addEnabled = False
    self.inputSelector_ct.removeEnabled = False
    self.inputSelector_ct.noneEnabled = False
    self.inputSelector_ct.showHidden = False
    self.inputSelector_ct.showChildNodeTypes = False
    self.inputSelector_ct.setMRMLScene( slicer.mrmlScene )
    self.inputSelector_ct.setToolTip( "Pick CT image" )
    inputsFormLayout.addRow("CT Volume: ", self.inputSelector_ct)

     #
    # output clustering volume selector
    #
    self.outputInfectClusterSelector = slicer.qMRMLNodeComboBox()
    self.outputInfectClusterSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.outputInfectClusterSelector.selectNodeUponCreation = True
    self.outputInfectClusterSelector.addEnabled = True
    self.outputInfectClusterSelector.removeEnabled = True
    self.outputInfectClusterSelector.noneEnabled = True
    self.outputInfectClusterSelector.showHidden = False
    self.outputInfectClusterSelector.showChildNodeTypes = False
    self.outputInfectClusterSelector.setMRMLScene( slicer.mrmlScene )
    self.outputInfectClusterSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Infection Label Map: ", self.outputInfectClusterSelector)

    #
    # output volume selector
    #
    self.outputLungClusterSelector = slicer.qMRMLNodeComboBox()
    self.outputLungClusterSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.outputLungClusterSelector.selectNodeUponCreation = True
    self.outputLungClusterSelector.addEnabled = True
    self.outputLungClusterSelector.removeEnabled = True
    self.outputLungClusterSelector.noneEnabled = True
    self.outputLungClusterSelector.showHidden = False
    self.outputLungClusterSelector.showChildNodeTypes = False
    self.outputLungClusterSelector.setMRMLScene( slicer.mrmlScene )
    self.outputLungClusterSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Lung Label Map: ", self.outputLungClusterSelector)

    #
    # output infection seg selector
    #
    self.segInfectSelector = slicer.qMRMLNodeComboBox()
    self.segInfectSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segInfectSelector.selectNodeUponCreation = True
    self.segInfectSelector.addEnabled = True
    self.segInfectSelector.removeEnabled = True
    self.segInfectSelector.noneEnabled = True
    self.segInfectSelector.showHidden = False
    self.segInfectSelector.showChildNodeTypes = False
    self.segInfectSelector.setMRMLScene( slicer.mrmlScene )
    self.segInfectSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Infection Segmentation: ", self.segInfectSelector)

    #
    # output lung seg selector
    #
    self.segLungSelector = slicer.qMRMLNodeComboBox()
    self.segLungSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segLungSelector.selectNodeUponCreation = True
    self.segLungSelector.addEnabled = True
    self.segLungSelector.removeEnabled = True
    self.segLungSelector.noneEnabled = True
    self.segLungSelector.showHidden = False
    self.segLungSelector.showChildNodeTypes = False
    self.segLungSelector.setMRMLScene( slicer.mrmlScene )
    self.segLungSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Lung Segmentation: ", self.segLungSelector)
    
    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    inputsFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    self.layout.addWidget(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector_ct.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputLungClusterSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputInfectClusterSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.segInfectSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.segLungSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector_ct.currentNode() and self.outputLungClusterSelector.currentNode() and self.segInfectSelector.currentNode() and self.segLungSelector.currentNode() and self.outputInfectClusterSelector.currentNode()

  def onApplyButton(self):
    logic = ConvNetCovidLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    logic.run(self.inputSelector_ct.currentNode(), self.outputLungClusterSelector.currentNode(), self.outputInfectClusterSelector.currentNode(), self.segInfectSelector.currentNode(), self.segLungSelector.currentNode(), enableScreenshotsFlag)


#
# ConvNetCovidLogic
#

class ConvNetCovidLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode_ct , outputLungClusterNode, outputInfectClusterNode, outputInfectSegmentationNode, outputLungSegmentationNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode_ct:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputLungClusterNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if not outputInfectSegmentationNode:
      logging.debug('isValidInputOutputData failed: no output segmentation node defined')
      return False
    if not outputLungSegmentationNode:
      logging.debug('isValidInputOutputData failed: no output segmentation node defined')
      return False
    if not outputInfectClusterNode:
      logging.debug('isValidInputOutputData failed: no output cluster node defined')
      return False
    #if inputVolumeNode_spect.GetID()==outputLesVolumeNode.GetID():
     # logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      #return False
    return True

  def pre_proc(self, img_ct):
    img_ct = np.rot90(img_ct, 1, (1,2))
    img_out = np.zeros_like(img_ct)
    for i in range(img_ct.shape[0]):
      img = img_ct[i,...]
      img = (img - img.min()) / (img.max() - img.min()+1e-6)
      #val = filters.threshold_otsu(img)
      tmp_img = img*255
      tmp_img = tmp_img.astype('uint8')
      #print(tmp_img.shape)
      val,_ = cv2.threshold(tmp_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      
      '''
      mask = np.zeros_like(img)
      mask[tmp_img > val] = 1
      mask = ndimage.binary_erosion(mask).astype(mask.dtype)
      labels_mask = measure.label(mask)
      regions = measure.regionprops(labels_mask)
      regions.sort(key=lambda x: x.area, reverse=True)
      if len(regions) > 1:
        for rg in regions[1:]:
          labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
      labels_mask[labels_mask != 0] = 1
      mask = labels_mask
      '''
      mask = np.zeros_like(img)
      mask[tmp_img > val] = 1
      
      mask = mask.astype('uint8')
      mask = ndimage.binary_erosion(mask).astype(mask.dtype)
      nb_components, out, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
      print(mask)
      max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
      mask = np.zeros_like(img)
      mask[out!=max_label] = 0
      mask[out==max_label] = 1
      mask = ndimage.binary_dilation(mask).astype(mask.dtype)
      mask = ndimage.binary_fill_holes(mask).astype(mask.dtype)
      mask[mask > 0] = 1
      
      img_out[i, :, :] = mask * img
    return img_out

  def resize_img(self, imgs, order=1, sz=(256, 256)):
    sz_x, sz_y = sz
    img_out = np.zeros((imgs.shape[0],sz_x, sz_y))
    for i in range(imgs.shape[0]):
        img = imgs[i,...]
        img_out[i,:,:] = cv2.resize(img, (sz_x, sz_y), interpolation = cv2.INTER_LINEAR)
    return img_out

  def resize_seg(self, imgs, order=0, sz=(256, 256)):
    sz_x, sz_y = sz
    img_out = np.zeros((imgs.shape[0],sz_x, sz_y))
    for i in range(imgs.shape[0]):
        img = imgs[i,...]
        img_out[i,:,:] = cv2.resize(img, (sz_x, sz_y),interpolation = cv2.INTER_NEAREST)
    return img_out

  def run(self, inputVolume_ct,outputLungClusterVolume, outputInfectClusterVolume,outputInfectSegmentation, outputLungSegmentation, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    outputInfectSegmentation.GetSegmentation().RemoveAllSegments()
    outputLungSegmentation.GetSegmentation().RemoveAllSegments()
    if not self.isValidInputOutputData(inputVolume_ct, outputLungClusterVolume, outputInfectClusterVolume, outputInfectSegmentation, outputLungSegmentation):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

     # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume_ct.GetID(), 'OutputVolume': outputLungClusterVolume.GetID(), 'ThresholdValue' : 0.2, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)
    cliParams = {'InputVolume': inputVolume_ct.GetID(), 'OutputVolume': outputInfectClusterVolume.GetID(), 'ThresholdValue' : 0.2, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    sz_x, sz_y = (256, 256)
    # convert volume to nd array
    ct_img = list(slicer.util.arrayFromVolume(inputVolume_ct))
    ct_img_org = np.copy(np.asarray(ct_img))
    ct_sz_x, ct_sz_y, ct_sz_z = ct_img_org.shape
    ct_img = self.pre_proc(ct_img_org)
    ct_img = self.resize_img(ct_img, order=1, sz=(sz_x, sz_y))
    vol_size = inputVolume_ct.GetImageData().GetDimensions()
    vol_size = np.asarray(vol_size)
    vol_center = vol_size/2
    
    lung_segout = np.zeros_like(ct_img)
    infect_segout = np.zeros_like(ct_img)
    for idx in range(ct_img.shape[0]):
      img = ct_img[idx,...].reshape(1, sz_x, sz_y, 1)
      img = (img - img.min()) / (img.max() - img.min()+1e-6)
      seg_out = ConvNetCovidWidget.segmentation_model.predict(img)
      seg_out = seg_out[0]
      seg_out = np.argmax(seg_out,axis=-1).reshape(sz_x*sz_y)
      img_seg = to_categorical(seg_out, 3).reshape(sz_x, sz_y, 3)
      infect_segout[idx,...] = img_seg[:, :, 2]
      lung_segout[idx,...]   = img_seg[:, :, 1]
    infect_segout = np.rot90(infect_segout, -1, (1,2))
    lung_segout = np.rot90(lung_segout, -1, (1,2))
    infect_segout = self.resize_seg(infect_segout, order=0, sz=(ct_sz_y, ct_sz_z))*2
    lung_segout = self.resize_seg(lung_segout, order=0, sz=(ct_sz_y, ct_sz_z))
    print(infect_segout.shape)
    print(np.sum(lung_segout))
    
    slicer.util.updateVolumeFromArray(outputInfectClusterVolume,infect_segout) # clustering results
    slicer.util.updateVolumeFromArray(outputLungClusterVolume,lung_segout)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputInfectClusterVolume, outputInfectSegmentation)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputLungClusterVolume, outputLungSegmentation) 


    
    

      




    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('ConvNetCovidTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class ConvNetCovidTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_ConvNetCovid1()

  def test_ConvNetCovid1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = ConvNetCovidLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
