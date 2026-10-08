import Foundation

enum BlenderMaterialScript {
    static func write(to folder: URL, settings: TextureSettings) throws {
        // Paths are derived from __file__, so moving the entire export folder stays safe.
        let script = """
        # Run in Blender's Scripting workspace. Creates a material without modifying geometry.
        import bpy
        from pathlib import Path
        # Choose one representation of this relief. Both maps describe the same height.
        USE_GEOMETRIC_DISPLACEMENT = False
        root = Path(__file__).resolve().parent
        material = bpy.data.materials.new(root.name)
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        nodes.clear()
        surface = nodes.new('ShaderNodeBsdfPrincipled')
        surface.location = (350, 150)
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (650, 150)
        links.new(surface.outputs['BSDF'], output.inputs['Surface'])
        def image_node(filename, color_space, y):
            image = bpy.data.images.load(str(root / filename), check_existing=False)
            image.colorspace_settings.name = color_space
            node = nodes.new('ShaderNodeTexImage')
            node.image = image
            node.location = (-650, y)
            return node
        diffuse = image_node('diffuse.png', 'sRGB', 500)
        roughness = image_node('roughness.exr', 'Non-Color', 200)
        normal_image = image_node('normal.exr', 'Non-Color', -100)
        height = image_node('displacement.exr', 'Non-Color', -400)
        links.new(diffuse.outputs['Color'], surface.inputs['Base Color'])
        links.new(roughness.outputs['Color'], surface.inputs['Roughness'])
        normal = nodes.new('ShaderNodeNormalMap')
        normal.space = 'TANGENT'
        normal.inputs['Strength'].default_value = 0.0 if USE_GEOMETRIC_DISPLACEMENT else 1.0
        normal.location = (-200, -100)
        links.new(normal_image.outputs['Color'], normal.inputs['Color'])
        links.new(normal.outputs['Normal'], surface.inputs['Normal'])
        displacement = nodes.new('ShaderNodeDisplacement')
        displacement.location = (350, -350)
        displacement.inputs['Midlevel'].default_value = 0.5
        displacement.inputs['Scale'].default_value = \(settings.displacementScaleMeters)
        links.new(height.outputs['Color'], displacement.inputs['Height'])
        if USE_GEOMETRIC_DISPLACEMENT:
            links.new(displacement.outputs['Displacement'], output.inputs['Displacement'])
        # Assign manually. Set Cycles displacement and subdivide your mesh when appropriate.
        print('Created material:', material.name)
        """
        try script.write(to: folder.appendingPathComponent("blender_material.py"), atomically: true, encoding: .utf8)
        let help = """
        Texture Studio material

        diffuse.png: 8-bit sRGB color. Roughness/displacement: linear scalar data.
        normal.exr: tangent-space OpenGL (+Y), encoded 0..1; set Blender Image Texture to Non-Color.
        All maps use the same crop and UV layout. Displacement is relative height, midlevel 0.5.
        Suggested material width: \(settings.materialWidthMeters) meters.
        Suggested displacement scale: \(settings.displacementScaleMeters) meters.

        Run blender_material.py in Blender's Scripting workspace to create a material.
        The default uses the OpenGL normal map. For geometry displacement, set
        USE_GEOMETRIC_DISPLACEMENT = True in the script, use Cycles displacement settings,
        and subdivide the mesh. That mode disables the equivalent full-height normal strength
        so the same relief is not applied twice. Assign the material to your object manually.
        These maps are artistic estimates from the original
        photo. Diffuse lighting correction cannot recover clipped highlights or fully hidden detail.
        Texture boundaries are not guaranteed seamless; inspect tiling before production use.
        """
        try help.write(to: folder.appendingPathComponent("BLENDER.txt"), atomically: true, encoding: .utf8)
    }
}
