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

class ViewController: UIViewController {
    @IBOutlet weak var preview: UIImageView!
    
    var model: OpenFace!
    var session = AVCaptureSession()
    var requests = [VNRequest]()
    var currentPixelBuffer: CVPixelBuffer?
    var count = 0
    var currentLabelRect: [CGRect]?
    var labelsArray: [String]?
    var repsMatrix: Matrix<Double>?
    
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
        readDataFromCSV()
        startLiveVideo()
        startFaceDetection()
//        testPerfomance()
    }
    
    func startLiveVideo() {
        //1
        session.sessionPreset = AVCaptureSession.Preset.hd4K3840x2160
        let captureDevice = AVCaptureDevice.default(for: AVMediaType.video)
        
        //2
        let deviceInput = try! AVCaptureDeviceInput(device: captureDevice!)
        let deviceOutput = AVCaptureVideoDataOutput()
        deviceOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        deviceOutput.setSampleBufferDelegate(self as AVCaptureVideoDataOutputSampleBufferDelegate, queue: DispatchQueue.global(qos: DispatchQoS.QoSClass.default))
        if let connection = deviceOutput.connection(with:  AVFoundation.AVMediaType.video) {
            guard connection.isVideoOrientationSupported else { return }
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
    
    func scale(rect: CGRect, view: UIView) -> CGRect {
        let x = rect.minX * view.frame.size.width
        let w = rect.width * view.frame.size.width
        let h = rect.height * view.frame.size.height
        let y = view.frame.size.height * (1 - rect.minY) - h
        return CGRect(x: x, y: y, width: w, height: h)
    }
    
    func detectFaceHandler(request: VNRequest, error: Error?) {
//        print("Complete handler", self.count)
        guard let observations = request.results as? [VNFaceObservation] else {
            print("no result")
            return
        }
//        if (observations.count == 0) { return }
//        print("number of faces", observations.count)
        self.currentLabelRect = []
        DispatchQueue.main.async() {
            self.preview.subviews.forEach({ $0.removeFromSuperview() })
        }
        let cropAndResizeFaceQueue = DispatchQueue(label: "com.wangderland.cropAndResizeQueue", qos: .userInteractive)
        for (idx, region) in observations.enumerated() {
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
                    print("First", self.count)
                    let scaledRect = self.scale(rect: boundingRect, view: self.preview)
                    self.currentLabelRect?.append(CGRect(x: scaledRect.minX, y: scaledRect.minY - 100, width: scaledRect.width * 2, height: scaledRect.height))
                    try MLRequestHandler.perform([self.MLRequest])
                } catch {
                    print(error)
                }
            }
        }
        DispatchQueue.main.async() {
            print("Second", self.count)
            self.preview.layer.sublayers?.removeSubrange(1...)
            let a = self.preview.subviews.count
//            print("a", a)
//            for view in self.preview.subviews {
//                view.removeFromSuperview()
//            }
            let b = self.preview.subviews.count
//            print("b", b)
            for region in observations {
                self.highlightFace(faceObservation: region)
            }
        }
    }
    
    func highlightFace(faceObservation: VNFaceObservation) {
        let boundingRect = faceObservation.boundingBox
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

    func buffer2Array<T>(length: Int, data: UnsafeMutableRawPointer, _: T.Type) -> [T] {
        let ptr = data.bindMemory(to: T.self, capacity: length)
        let buffer = UnsafeBufferPointer(start: ptr, count: length)
        return Array(buffer)
    }
    
    func genEmbeddingsHandler(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [ VNCoreMLFeatureValueObservation] else { return }
        observations.forEach { observe in
            guard let emb = observe.featureValue.multiArrayValue else { return }
            let doubleValueEmb = buffer2Array(length: emb.count, data: emb.dataPointer, Double.self)
            guard let repsMatrix = self.repsMatrix else { return }
            let embMatrix = Matrix(Array(repeating: doubleValueEmb, count: repsMatrix.rows))
            let diff = repsMatrix - embMatrix
            let squredDiff = myPow(diff, 2)
            let l2 = sum(squredDiff, axies:.row)
            let minVal = l2.grid.min()
            var ans: String = "Unknown"
            guard let minIdx = l2.grid.index(of: minVal!) else { return }
            guard let labelsArray = self.labelsArray else { return }
            ans = labelsArray[minIdx]
//            print("current idx", self.count)
//            print("My ans:", ans)
            DispatchQueue.main.async() {
                guard let labelRectArr = self.currentLabelRect else { return }
                for labelRect in labelRectArr {
                    let faceLabel = UILabel()
                    faceLabel.backgroundColor = UIColor.lightGray
                    faceLabel.textColor = UIColor.black
                    faceLabel.frame = labelRect
                    faceLabel.text = ans
                    let a = self.preview.subviews.count
                    print("Third", self.count)
                    //                print("gem A", a)
                    self.preview.addSubview(faceLabel)
                    let b = self.preview.subviews.count
                    //                print("gem B", b)
                }
            }
        }
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
    
    func readDataFromCSV() {
        guard let labelsPath = Bundle.main.path(forResource: "labels_mini", ofType: "csv") else { return }
        guard let repsPath = Bundle.main.path(forResource: "reps_mini", ofType: "csv") else { return }
        let labels = try! String(contentsOfFile: labelsPath, encoding: String.Encoding.utf8)
        let reps = try! String(contentsOfFile: repsPath, encoding: String.Encoding.utf8)
        let labelsArray: [String] = labels.components(separatedBy: "\r").map{ $0.components(separatedBy: "/")[3] }
        let repsArray: [[Double]] = reps
            .components(separatedBy: "\r")
            .map{ $0.components(separatedBy: ",").map{ Double($0)! }}
        let repsMatrix = Matrix(repsArray)
        self.labelsArray = labelsArray
        self.repsMatrix = repsMatrix
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
//            print("try perform request", self.count)
            self.currentPixelBuffer = pixelBuffer
            try imageRequestHandler.perform(self.requests)
            self.count += 1
        } catch {
            print(error)
        }
    }
}

