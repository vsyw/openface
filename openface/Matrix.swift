//
//  Matrix.swift
//  openface
//
//  Created by victor.sy_wang on 2017/10/17.
//  Copyright © 2017年 victor. All rights reserved.
//

import Foundation
import Accelerate

public enum MatrixAxies {
    case row
    case column
}

public struct Matrix<T> where T: FloatingPoint, T: ExpressibleByFloatLiteral {
    public typealias Element = T
    
    let rows: Int
    let columns: Int
    var grid: [Element]
    
    public init(rows: Int, columns: Int, repeatedValue: Element) {
        self.rows = rows
        self.columns = columns
        
        self.grid = [Element](repeating: repeatedValue, count: rows * columns)
    }
    
    public init(_ contents: [[Element]]) {
        let m: Int = contents.count
        let n: Int = contents[0].count
        let repeatedValue: Element = 0.0
        
        self.init(rows: m, columns: n, repeatedValue: repeatedValue)
        
        for (i, row) in contents.enumerated() {
            grid.replaceSubrange(i*n..<i*n+Swift.min(m, row.count), with: row)
        }
    }
    
    public init(rows: Int, columns: Int, valueFunc: ()->Element){
        self.rows = rows
        self.columns = columns
        self.grid = [Element](repeating: valueFunc(), count: rows * columns)
        for i in 0..<grid.count{
            self.grid[i] = valueFunc()
        }
    }
    
    public subscript(row: Int, column: Int) -> Element {
        get {
            assert(indexIsValidForRow(row, column: column))
            return grid[(row * columns) + column]
        }
        
        set {
            assert(indexIsValidForRow(row, column: column))
            grid[(row * columns) + column] = newValue
        }
    }
    
    public subscript(row row: Int) -> [Element] {
        get {
            assert(row < rows)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            return Array(grid[startIndex..<endIndex])
        }
        
        set {
            assert(row < rows)
            assert(newValue.count == columns)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            grid.replaceSubrange(startIndex..<endIndex, with: newValue)
        }
    }
    
    public subscript(column column: Int) -> [Element] {
        get {
            var result = [Element](repeating: 0.0, count: rows)
            for i in 0..<rows {
                let index = i * columns + column
                result[i] = self.grid[index]
            }
            return result
        }
        
        set {
            assert(column < columns)
            assert(newValue.count == rows)
            for i in 0..<rows {
                let index = i * columns + column
                grid[index] = newValue[i]
            }
        }
    }
    
    fileprivate func indexIsValidForRow(_ row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
}

// MARK: - Printable
extension Matrix: CustomStringConvertible {
    public var description: String {
        var description = ""
        
        for i in 0..<rows {
            let contents = (0..<columns).map{"\(self[i, $0])"}.joined(separator: "\t")
            
            switch (i, rows) {
            case (0, 1):
                description += "(\t\(contents)\t)"
            case (0, _):
                description += "⎛\t\(contents)\t⎞"
            case (rows - 1, _):
                description += "⎝\t\(contents)\t⎠"
            default:
                description += "⎜\t\(contents)\t⎥"
            }
            
            description += "\n"
        }
        
        return description
    }
}

extension Matrix: Equatable {}
public func ==<T> (lhs: Matrix<T>, rhs: Matrix<T>) -> Bool {
    return lhs.rows == rhs.rows && lhs.columns == rhs.columns && lhs.grid == rhs.grid
}


// MARK: -
public func add(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
    
    var results = y
    cblas_saxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
    
    return results
}

public func add(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
    
    var results = y
    cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
    
    return results
}

public func sub(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
    
    var results = negate(y)
    cblas_saxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
    
    return results
}

public func sub(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
    
    var results = negate(y)
    cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
    
    return results
}
public func mul(_ alpha: Float, x: Matrix<Float>) -> Matrix<Float> {
    var results = x
    cblas_sscal(Int32(x.grid.count), alpha, &(results.grid), 1)
    
    return results
}

public func mul(_ alpha: Double, x: Matrix<Double>) -> Matrix<Double> {
    var results = x
    cblas_dscal(Int32(x.grid.count), alpha, &(results.grid), 1)
    
    return results
}

public func mul(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")
    
    var results = Matrix<Float>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))
    
    return results
}

public func mul(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")
    
    var results = Matrix<Double>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))
    
    return results
}

//public func elmul(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
//    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix must have the same dimensions")
//    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
//    result.grid = x.grid * y.grid
//    return result
//}
//
//public func elmul(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
//    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix must have the same dimensions")
//    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
//    result.grid = x.grid * y.grid
//    return result
//}
//
//public func div(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
//    let yInv = inv(y)
//    precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
//    return mul(x, y: yInv)
//}
//
//public func div(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
//    let yInv = inv(y)
//    precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
//    return mul(x, y: yInv)
//}

public func withUnsafePointersAndCountsTo<A: ContinuousCollection>(_ a: A, body: (UnsafePointer<A.Element>, Int) throws -> Void) rethrows {
    try a.withUnsafeBufferPointer { (a: UnsafeBufferPointer<A.Element>) throws -> Void in
        if let ab = a.baseAddress {
            try body(ab, a.count)
        }
    }
}

public func withUnsafePointersAndCountsTo<A: ContinuousCollection, B: ContinuousCollection>(_ a: A, _ b: B, body: (UnsafePointer<A.Element>, Int, UnsafePointer<B.Element>, Int) throws -> Void) rethrows {
    try a.withUnsafeBufferPointer { (a: UnsafeBufferPointer<A.Element>) throws -> Void in
        try b.withUnsafeBufferPointer { (b: UnsafeBufferPointer<B.Element>) throws -> Void in
            if let ab = a.baseAddress, let bb = b.baseAddress {
                try body(ab, a.count, bb, b.count)
            }
        }
    }
}


public protocol ContinuousCollection: Collection {
    /// Calls a closure with a pointer to the array's contiguous storage.
    ///
    /// The pointer passed as an argument to `body` is valid only during the execution of
    /// `withUnsafeBufferPointer(_:)`. Do not store or return the pointer for later use.
    ///
    /// - Parameter body: A closure with an `UnsafeBufferPointer` parameter that points to the contiguous storage for
    ///   the array.  If `body` has a return value, that value is also used as the return value for the
    ///   `withUnsafeBufferPointer(_:)` method. The pointer argument is valid only for the duration of the method's
    ///   execution.
    /// - Returns: The return value, if any, of the `body` closure parameter.
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R
}

public protocol ContinuousMutableCollection: ContinuousCollection {
    /// Calls the given closure with a pointer to the array's mutable contiguous storage.
    ///
    /// The pointer passed as an argument to `body` is valid only during the execution of
    /// `withUnsafeMutableBufferPointer(_:)`. Do not store or return the pointer for later use.
    ///
    /// - Parameter body: A closure with an `UnsafeMutableBufferPointer` parameter that points to the contiguous
    ///   storage for the array. If `body` has a return value, that value is also used as the return value for the
    ///   `withUnsafeMutableBufferPointer(_:)` method. The pointer argument is valid only for the duration of the
    ///   method's execution.
    /// - Returns: The return value, if any, of the `body` closure parameter.
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R
}

extension Array: ContinuousMutableCollection {}
extension ContiguousArray: ContinuousMutableCollection {}
extension ArraySlice: ContinuousMutableCollection {}

public func pow<X: ContinuousCollection, Y: ContinuousCollection>(_ x: X, _ y: Y) -> [Double] where X.Iterator.Element == Double, Y.Iterator.Element == Double {
    var results = [Double](repeating: 0.0, count: numericCast(x.count))
    results.withUnsafeMutableBufferPointer { pointer in
        withUnsafePointersAndCountsTo(x, y) { xp, xc, yp, _ in
            vvpow(pointer.baseAddress!, xp, yp, [Int32(xc)])
        }
    }
    return results
}

public func pow<X: ContinuousCollection>(_ x: X, _ y: Double) -> [Double] where X.Iterator.Element == Double {
    let yVec = [Double](repeating: y, count: numericCast(x.count))
    return pow(yVec, x)
}

public func myPow(_ x: Matrix<Double>, _ y: Double) -> Matrix<Double> {
    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = pow(x.grid, y)
    return result
}
////
//public func pow(_ x: Matrix<Float>, _ y: Float) -> Matrix<Float> {
//    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
//    result.grid = pow(x.grid, y)
//    return result
//}
//
//public func exp(_ x: Matrix<Double>) -> Matrix<Double> {
//    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
//    result.grid = exp(x.grid)
//    return result
//}
//
//public func exp(_ x: Matrix<Float>) -> Matrix<Float> {
//    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
//    result.grid = exp(x.grid)
//    return result
//}
//

public func sum<C: ContinuousCollection>(_ x: C) -> Double where C.Iterator.Element == Double {
    var result: Double = 0.0
    withUnsafePointersAndCountsTo(x) { x, count in
        withUnsafeMutablePointer(to: &result) { pointer in
            vDSP_sveD(x, 1, pointer, vDSP_Length(count))
        }
    }
    return result
}

public func sum(_ x: Matrix<Double>, axies: MatrixAxies = .column) -> Matrix<Double> {
    
    switch axies {
    case .column:
        var result = Matrix<Double>(rows: 1, columns: x.columns, repeatedValue: 0.0)
        for i in 0..<x.columns {
            result.grid[i] = sum(x[column: i])
        }
        return result
        
    case .row:
        var result = Matrix<Double>(rows: x.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<x.rows {
            result.grid[i] = sum(x[row: i])
        }
        return result
    }
}

public func negate(_ x: Matrix<Float>) -> Matrix<Float> {
    var results = x
    vDSP_vneg(x.grid, 1, &(results.grid), 1, vDSP_Length(results.grid.count))
    
    return results
}

public func negate(_ x: Matrix<Double>) -> Matrix<Double> {
    var results = x
    vDSP_vnegD(x.grid, 1, &(results.grid), 1, vDSP_Length(results.grid.count))
    return results
}

public func transpose(_ x: Matrix<Float>) -> Matrix<Float> {
    var results = Matrix<Float>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtrans(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))
    
    return results
}

public func transpose(_ x: Matrix<Double>) -> Matrix<Double> {
    var results = Matrix<Double>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtransD(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))
    
    return results
}

// MARK: - Operators
public func + (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return add(lhs, y: rhs)
}

public func + (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return add(lhs, y: rhs)
}

public func - (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return sub(lhs, y: rhs)
}

public func - (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return sub(lhs, y: rhs)
}


public func * (lhs: Float, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, x: rhs)
}

public func * (lhs: Double, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, x: rhs)
}

public func * (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, y: rhs)
}

public func * (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, y: rhs)
}

postfix operator ′
public postfix func ′ (value: Matrix<Float>) -> Matrix<Float> {
    return transpose(value)
}

public postfix func ′ (value: Matrix<Double>) -> Matrix<Double> {
    return transpose(value)
}
