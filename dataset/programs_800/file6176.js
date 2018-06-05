'use strict'

var util = require('util'),
    geo = require('geometry'),
    ccp = geo.ccp

/**
 * @class
 * Represents a single frame of animation for a cocos.Sprite
 *
 * A SpriteFrame has:
 * - texture: A Texture2D that will be used by the Sprite
 * - rectangle: A rectangle of the texture
 *
 * @example
 * var frame = new SpriteFrame({texture: texture, rect: rect})
 * sprite.displayFrame = frame
 *
 * @memberOf cocos
 *
 * @opt {cocos.Texture2D} texture The texture to draw this frame using
 * @opt {geometry.Rect} rect The rectangle inside the texture to draw
 */
function SpriteFrame (opts) {
    SpriteFrame.superclass.constructor.call(this, opts)

    this.texture      = opts.texture
    this.rect         = opts.rect
    this.rotated      = !!opts.rotate
    this.offset       = opts.offset || ccp(0, 0)
    this.originalSize = opts.originalSize || util.copy(this.rect.size)
}

SpriteFrame.inherit(Object, /** @lends cocos.SpriteFrame# */ {
    rect: null,
    rotated: false,
    offset: null,
    originalSize: null,
    texture: null,

    /**
     * @ignore
     */
    toString: function () {
        return "[object SpriteFrame | TextureName=" + this.texture.name + ", Rect = (" + this.rect.origin.x + ", " + this.rect.origin.y + ", " + this.rect.size.width + ", " + this.rect.size.height + ")]"
    },

    /**
     * Make a copy of this frame
     *
     * @returns {cocos.SpriteFrame} Exact copy of this object
     */
    copy: function () {
        return new SpriteFrame({rect: this.rect, rotated: this.rotated, offset: this.offset, originalSize: this.originalSize, texture: this.texture})
    }

})

exports.SpriteFrame = SpriteFrame

// vim:et:st=4:fdm=marker:fdl=0:fdc=1
