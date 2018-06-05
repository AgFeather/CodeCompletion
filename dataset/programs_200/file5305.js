/* globals L */

var util = require('../../util/util');

var ArcGisServerDynamicLayer = L.Class.extend({
  includes: [
    L.Mixin.Events,
    require('../../mixin/esri')
  ],
  options: {
    opacity: 1,
    position: 'front'
  },
  _defaultLayerParams: {
    bboxSR: 3857,
    f: 'image',
    format: 'png24',
    imageSR: 3857,
    layers: '',
    transparent: true
  },
  initialize: function(options) {
    util.strict(options.url, 'string');

    this._layerParams = L.Util.extend({}, this._defaultLayerParams);
    this._serviceUrl = this._cleanUrl(options.url);

    for (var option in options) {
      if (this._defaultLayerParams.hasOwnProperty(option)) {
        this._layerParams[option] = options[option];
      }
    }

    this._parseLayers();
    L.Util.setOptions(this, options);

    if (!this._layerParams.transparent) {
      this.options.opacity = 1;
    }

    if (options.clickable === false) {
      this._hasInteractivity = false;
    }

    this._getMetadata();
  },
  onAdd: function(map) {
    this._map = map;
    this._moveHandler = this._debounce(this._update, 150, this);

    if (map.options.crs && map.options.crs.code) {
      var sr = map.options.crs.code.split(':')[1];

      this._layerParams.bboxSR = sr;
      this._layerParams.imageSR = sr;
    }

    map.on('moveend', this._moveHandler, this);
    this._update();
  },
  onRemove: function(map) {
    if (this._currentImage) {
      this._map.removeLayer(this._currentImage);
    }

    map.off('moveend', this._moveHandler, this);
  },
  _debounce: function(fn, delay) {
    var timer = null;

    return function() {
      var context = this || context, args = arguments;

      clearTimeout(timer);

      timer = setTimeout(function () {
        fn.apply(context, args);
      }, delay);
    };
  },
  _getImageUrl: function () {
    var map = this._map,
      bounds = map.getBounds(),
      crs = map.options.crs,
      layerParams = this._layerParams,
      size = map.getSize(),
      ne = crs.project(bounds._northEast),
      options = this.options,
      sw = crs.project(bounds._southWest);

    layerParams.bbox = [sw.x, sw.y, ne.x, ne.y].join(',');
    layerParams.size = size.x + ',' + size.y;

    if (options.edit) {
      layerParams.nocache = new Date().getTime();
    }

    if (options.token) {
      layerParams.token = options.token;
    }

    return this._serviceUrl + 'export' + L.Util.getParamString(layerParams);
  },
  _parseLayers: function () {
    if (typeof this._layerParams.layers === 'undefined') {
      delete this._layerParams.layerOption;
      return;
    }

    var action = this._layerParams.layerOption || null,
      layers = this._layerParams.layers || null,
      verb = 'show',
      verbs = ['exclude', 'hide', 'include', 'show'];

    delete this._layerParams.layerOption;

    if (!action) {
      if (L.Util.isArray(layers)) {
        this._layerParams.layers = verb + ':' + layers.join(',');
      } else if (typeof layers === 'string') {
        var match = layers.match(':');

        if (match) {
          layers = layers.split(match[0]);

          if (Number(layers[1].split(',')[0])) {
            if (verbs.indexOf(layers[0]) !== -1) {
              verb = layers[0];
            }

            layers = layers[1];
          }
        }

        this._layerParams.layers = verb + ':' + layers;
      }
    } else {
      if (verbs.indexOf(action) !== -1) {
        verb = action;
      }

      this._layerParams.layers = verb + ':' + layers;
    }
  },
  _update: function() {
    var bounds, image, zoom;

    if (this._animatingZoom || (this._map._panTransition && this._map._panTransition._inProgress)) {
      return;
    }

    zoom = this._map.getZoom();

    if (zoom > this.options.maxZoom || zoom < this.options.minZoom) {
      return;
    }

    bounds = this._map.getBounds();
    bounds._southWest.wrap();
    bounds._northEast.wrap();
    image = new L.ImageOverlay(this._getImageUrl(), bounds, {
      opacity: 0
    }).addTo(this._map);
    image.on('load', function(e){
      var newImage = e.target,
        oldImage = this._currentImage;

      if (newImage._bounds.equals(bounds)) {
        this._currentImage = newImage;

        if (this.options.position === 'front') {
          this._currentImage.bringToFront();
        } else {
          this._currentImage.bringToBack();
        }

        this._currentImage.setOpacity(this.options.opacity);

        if (oldImage) {
          this._map.removeLayer(oldImage);
        }
      } else {
        this._map.removeLayer(newImage);
      }
    }, this);
    this.fire('loading', {
      bounds: bounds
    });
  },
  bringToBack: function(){
    this.options.position = 'back';
    this._currentImage.bringToBack();
    return this;
  },
  bringToFront: function(){
    this.options.position = 'front';
    this._currentImage.bringToFront();
    return this;
  },
  redraw: function() {
    this._update();
  },
  setLayers: function(layers) {
    if (typeof layers === 'number') {
      layers = layers.toString();
    }

    this._layerParams.layers = layers;
    this._parseLayers();
    this._map.removeLayer(this._currentImage);
    this._update();
  },
  setOpacity: function(opacity) {
    this.options.opacity = opacity;
    this._currentImage.setOpacity(opacity);
  }
});

module.exports = function(options) {
  options = options || {};

  if (!options.type) {
    options.type = 'arcgisserver';
  }

  return new ArcGisServerDynamicLayer(options);
};
