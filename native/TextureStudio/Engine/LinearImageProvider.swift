import Foundation
import CoreImage

/// A tiled colour-management boundary. Colour processing is rendered explicitly into linear
/// float RGBA before numeric maps see it, regardless of the eventual EXR render context.
/// This preserves wide-gamut import while preventing a map exporter from reinterpreting sRGB.
final class LinearImageProvider: NSObject {
    private let image: CIImage
    private let context: CIContext
    private let height: Int
    private let colorSpace = CGColorSpace(name:CGColorSpace.extendedLinearSRGB)!

    init(image: CIImage, context: CIContext) {
        self.image = image
        self.context = context
        self.height = Int(image.extent.height)
    }

    @objc(provideImageData:bytesPerRow:origin::size::userInfo:)
    override func provideImageData(_ data: UnsafeMutableRawPointer, bytesPerRow rowbytes: Int,
                          origin originx: Int, _ originy: Int, size width: Int, _ tileHeight: Int,
                          userInfo: Any?) {
        let bounds = CGRect(x:originx,y:height-originy-tileHeight,width:width,height:tileHeight)
        context.render(image,toBitmap:data,rowBytes:rowbytes,bounds:bounds,format:.RGBAf,colorSpace:colorSpace)
    }

    static func image(_ image: CIImage, context: CIContext) -> CIImage {
        CIImage(imageProvider:LinearImageProvider(image:image,context:context),
                size:Int(image.extent.width),Int(image.extent.height),format:.RGBAf,colorSpace:nil,
                options:[.providerTileSize:256])
    }
}
