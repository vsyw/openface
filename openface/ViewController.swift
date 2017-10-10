//
//  ViewController.swift
//  openface
//
//  Created by victor.sy_wang on 2017/9/17.
//  Copyright © 2017年 victor. All rights reserved.
//

import UIKit
import CoreML
import AVFoundation
import Vision
import Accelerate

func pixelBufferFromImage(image: UIImage) -> CVPixelBuffer {
    
//    let newImage = resize(image: image, newSize: CGSize(width: 224/3.0, height: 224/3.0))
    
    let ciimage = CIImage(image: image)
    let tmpcontext = CIContext(options: nil)
    let cgimage =  tmpcontext.createCGImage(ciimage!, from: ciimage!.extent)
    
    let cfnumPointer = UnsafeMutablePointer<UnsafeRawPointer>.allocate(capacity: 1)
    let cfnum = CFNumberCreate(kCFAllocatorDefault, .intType, cfnumPointer)
    let keys: [CFString] = [kCVPixelBufferCGImageCompatibilityKey, kCVPixelBufferCGBitmapContextCompatibilityKey, kCVPixelBufferBytesPerRowAlignmentKey]
    let values: [CFTypeRef] = [kCFBooleanTrue, kCFBooleanTrue, cfnum!]
    let keysPointer = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
    let valuesPointer =  UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
    keysPointer.initialize(to: keys)
    valuesPointer.initialize(to: values)
    
    let options = CFDictionaryCreate(kCFAllocatorDefault, keysPointer, valuesPointer, keys.count, nil, nil)
    
    let width = cgimage!.width
    let height = cgimage!.height
    
    var pxbuffer: CVPixelBuffer?
    var status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                     kCVPixelFormatType_32BGRA, options, &pxbuffer)
    status = CVPixelBufferLockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0))
    
    let bufferAddress = CVPixelBufferGetBaseAddress(pxbuffer!)
    
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesperrow = CVPixelBufferGetBytesPerRow(pxbuffer!)
    let context = CGContext(data: bufferAddress,
                            width: width,
                            height: height,
                            bitsPerComponent: 8,
                            bytesPerRow: bytesperrow,
                            space: rgbColorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    context?.concatenate(CGAffineTransform(rotationAngle: 0))
//    context?.concatenate(__CGAffineTransformMake( 1, 0, 0, -1, 0, CGFloat(height) )) //Flip Vertical
    
    
    
    context?.draw(cgimage!, in: CGRect(x:0, y:0, width:CGFloat(width), height:CGFloat(height)))
    status = CVPixelBufferUnlockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0))
    return pxbuffer!
    
}


class ViewController: UIViewController {
    @IBOutlet weak var preview: UIImageView!
    
    var model: OpenFace!
    var session = AVCaptureSession()
    var requests = [VNRequest]()
    var currentPixelBuffer: CVPixelBuffer?
    var count = 0
    lazy var MLRequest: VNCoreMLRequest = {
        // Load the ML model through its generated class and create a Vision request for it.
        do {
            let model = try VNCoreMLModel(for: OpenFace().model)
            return VNCoreMLRequest(model: model, completionHandler: self.genEmbeddingsHandler)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        startLiveVideo()
        startFaceDetection()
//        generateEmbeddings()
    }
    
    func startLiveVideo() {
        //1
        session.sessionPreset = AVCaptureSession.Preset.hd1920x1080
        let captureDevice = AVCaptureDevice.default(for: AVMediaType.video)
        
        //2
        let deviceInput = try! AVCaptureDeviceInput(device: captureDevice!)
        let deviceOutput = AVCaptureVideoDataOutput()
        deviceOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        deviceOutput.setSampleBufferDelegate(self as AVCaptureVideoDataOutputSampleBufferDelegate, queue: DispatchQueue.global(qos: DispatchQoS.QoSClass.default))
        if let connection = deviceOutput.connection(with:  AVFoundation.AVMediaType.video) {
            guard connection.isVideoOrientationSupported else { return }
            print("Fuck youuuu")
            connection.videoOrientation = .portrait
        }
        session.addInput(deviceInput)
        session.addOutput(deviceOutput)
        
        //3
        let videoLayer = AVCaptureVideoPreviewLayer(session: session)
        videoLayer.frame = preview.bounds
        videoLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        preview.layer.addSublayer(videoLayer)
        
        session.startRunning()
    }

    override func viewDidLayoutSubviews() {
        preview.layer.sublayers?[0].frame = preview.bounds
    }
    
    func startFaceDetection() {
        let faceRequest = VNDetectFaceRectanglesRequest(completionHandler: self.detectFaceHandler)
        self.requests = [faceRequest]
    }
    
    func detectFaceHandler(request: VNRequest, error: Error?) {
        print("Complete handler", self.count)
        guard let observations = request.results as? [VNFaceObservation] else {
            print("no result")
            return
        }
        print("number of faces", observations.count)
        let cropAndResizeFaceQueue = DispatchQueue(label: "com.wangderland.cropAndResizeQueue", qos: .userInteractive)
        for region in observations {
            cropAndResizeFaceQueue.async {
                guard let pixelBuffer = self.currentPixelBuffer else { return }
                let boundingRect = region.boundingBox
                let x = boundingRect.minX * CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let w = boundingRect.width * CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let h = boundingRect.height * CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                let y = CGFloat(CVPixelBufferGetHeight(pixelBuffer)) * (1 - boundingRect.minY) - h
                let scaledRect = CGRect(x: x, y: y, width: w, height: h)
                guard let croppedPixelBuffer = self.cropFace(imageBuffer: pixelBuffer, region: scaledRect) else { return }
                let MLRequestHandler = VNImageRequestHandler(cvPixelBuffer: croppedPixelBuffer, orientation: CGImagePropertyOrientation(rawValue: 1)!, options: [:])
                do {
                    try MLRequestHandler.perform([self.MLRequest])
                } catch {
                    print(error)
                }
            }
        }
        
        DispatchQueue.main.async() {
            self.preview.layer.sublayers?.removeSubrange(1...)
            
            for region in observations {
                self.highlightFace(faceObservation: region)
            }
        }
    }
    
    func highlightFace(faceObservation: VNFaceObservation) {
        let boundingRect = faceObservation.boundingBox
        print("boundingRect", boundingRect)
        
        let x = boundingRect.minX * preview.frame.size.width
        let w = boundingRect.width * preview.frame.size.width
        let h = boundingRect.height * preview.frame.size.height
        let y = preview.frame.size.height * (1 - boundingRect.minY) - h
        let conv_rect = CGRect(x: x, y: y, width: w, height: h)
        
        let outline = CAShapeLayer()
        outline.frame = conv_rect
        outline.borderWidth = 1.0
        outline.borderColor = UIColor.blue.cgColor
        preview.layer.addSublayer(outline)
    }
    
    func cropFace(imageBuffer: CVPixelBuffer, region: CGRect) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(imageBuffer, .readOnly)
        let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
        // calculate start position
        let bytesPerPixel = 4
        let startAddress = baseAddress?.advanced(by: Int(region.minY) * bytesPerRow + Int(region.minX) * bytesPerPixel)
        var croppedImageBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault,
                                                  Int(region.width),
                                                  Int(region.height),
                                                  kCVPixelFormatType_32BGRA,
                                                  startAddress!,
                                                  bytesPerRow,
                                                  nil,
                                                  nil,
                                                  nil,
                                                  &croppedImageBuffer)
        CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)
        if (status != 0) {
            print("CVPixelBufferCreate Error: ", status)
        }
        return croppedImageBuffer
    }
    
    func cropFaceWithCGContext(imageBuffer: CVPixelBuffer, region: CGRect) {
        CVPixelBufferLockBaseAddress(imageBuffer, .readOnly)
        let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let startAddress = baseAddress?.advanced(by: Int(region.minY) * bytesPerRow + Int(region.minX) * bytesPerPixel)
        let context = CGContext(data: startAddress, width: Int(region.width), height: Int(region.height), bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)
        let _: CGImage = context!.makeImage()!
    }

    func genEmbeddingsHandler(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [ VNCoreMLFeatureValueObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        print("What's this shit", observations)
    }
    
    func testPerfomance() {
        if let sourceImage = UIImage(named: "Aaron_Eckhart_0001") {
            let imageBuffer = pixelBufferFromImage(image: sourceImage)
            print("cvpixelbuffer", imageBuffer)
            do {
                let start = CACurrentMediaTime()
                let emb = try model?.prediction(data: imageBuffer)
                let end = CACurrentMediaTime()
                print("Time - \(end - start)")
                print("fuck you", emb!.output)
            } catch {
            }
            
            var start = CACurrentMediaTime()
            let output = resizePixelBuffer(imageBuffer, cropX: 0, cropY: 0, cropWidth: 49, cropHeight: 49, scaleWidth: 96, scaleHeight: 96)!
            //            let output = cropFace(imageBuffer: imageBuffer, region: CGRect(x: 0, y: 0, width: 49, height: 49))
            var end = CACurrentMediaTime()
            print("CropFace Time:", end - start)
            start = CACurrentMediaTime()
            resizePixelBuffer(output, width: 96, height: 96, output: output, context: CIContext())
            end = CACurrentMediaTime()
            print("Resize Time: ", end - start)
            
            start = CACurrentMediaTime()
            cropFaceWithCGContext(imageBuffer: imageBuffer, region: CGRect(x: 0, y: 0, width: 49, height: 49))
            end = CACurrentMediaTime()
            print("CropFaceWithCGContext", end - start)
        }
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        connection.videoOrientation = AVCaptureVideoOrientation.portrait
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        var requestOptions:[VNImageOption : Any] = [:]
        
        if let camData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
            requestOptions = [.cameraIntrinsics:camData]
        }
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: CGImagePropertyOrientation(rawValue: 1)!, options: requestOptions)
        
        do {
            print("try perform request", self.count)
            self.currentPixelBuffer = pixelBuffer
            try imageRequestHandler.perform(self.requests)
            self.count += 1
        } catch {
            print(error)
        }
    }
}

