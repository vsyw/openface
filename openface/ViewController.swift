//
//  ViewController.swift
//  openface
//
//  Created by victor.sy_wang on 2017/9/17.
//  Copyright © 2017年 victor. All rights reserved.
//

import UIKit
import CoreML

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

    var model: OpenFace!
    
    @IBOutlet weak var preview: UIImageView!
    override func viewDidLoad() {
        super.viewDidLoad()
        generateEmbeddings()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func generateEmbeddings() {
        model = OpenFace()
        if let sourceImage = UIImage(named: "Aaron_Eckhart_0001") {
            preview.image = sourceImage
            let inputImage = pixelBufferFromImage(image: sourceImage)
            print("hello")
            do {
                let start = CACurrentMediaTime()
                let emb = try model?.prediction(data: inputImage)
                let end = CACurrentMediaTime()
                print("Time - \(end - start)")
                print("fuck you", emb!.output)
            } catch {

            }

        }
    }


}
