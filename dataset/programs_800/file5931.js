exports.ptype = 120; // must be unique per service, see dMap.js

exports.profile = {
  oembed: function(data) {
    var ret = {
      id            : data.guid,
      title         : data.givenName + ' ' + data.familyName,
      url           : data.profileUrl,
      thumbnail_url : data.image.imageUrl,
      provider_name : 'yahoo'
    };
    if (data.emails) {
      for (var emailObj in data.emails) {
        if (emailObj.primary) {
          ret.email = emailObj.handle;
        }
      }
    }
    return ret;
  }
};

exports.defaults = {
  self: 'profile'
};