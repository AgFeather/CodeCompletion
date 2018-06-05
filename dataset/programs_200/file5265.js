
/**
  Session
  
  Provides complete session management for applications.
  
  » Configuration Options
  
    {boolean} guestSessions: If set to true, will enable sessions for guest users
    {int} regenInterval: Interval to regenerate the sessionId (seconds)
    {int} permanentExpires: Permanent sessions timeout (seconds)
    {int} temporaryExpires: Temporary (browser) sessions timeout (seconds)
    {int} defaultExpires: Default number for browser sessions to live on the storage backend (seconds)
    {int} guestExpires: Guest sessions timeout (seconds)
    {array} typecastVars: Session properties to automatically typecast when loading session data
    {string} sessCookie: Default session cookie name
    {string} hashCookie: Default session hash name
    {string} salt: Salt used to generate session hashes
    {string|object} storage: Resource string pointing to the storage backend to use, or Storage instance.

 */

var app = protos.app;
var slice = Array.prototype.slice;
var moment = require('moment');

var _ = require('underscore');
var util = require('util');
    
require('./request.js');

function Session(config, middleware) {

  // Automatically load cookie_parser module dependency
  if (!app.supports.cookie_parser) {
    throw new Error("The 'session' middleware requires 'cookie_parser'.");
  }

  app[middleware] = this; // Attach to application singleton

  // Middleware configuration defaults
  this.config = config = _.extend({
    guestSessions: false,
    regenInterval: "5 minutes",
    permanentExpires: "1 month",
    defaultExpires: "1 day",
    temporaryExpires: "24 hours",
    guestExpires: "1 week",
    typecastVars: [],
    autoTypecast: true,
    sessCookie: "_sess",
    hashCookie: "_shash",
    salt: null
  }, config);
  
  // Parse time
  var timeOpts = ['regenInterval', 'permanentExpires', 'defaultExpires', 'temporaryExpires', 'guestExpires'];
  
  for (var key in config) {
    if (timeOpts.indexOf(key) >= 0 && typeof config[key] == 'string') {
      config[key] = time(config[key]);
    }
  }
  
  if (typeof config.salt != 'string') throw new Error("Session: you must specify a salt");

  switch (typeof config.storage) {
    case 'object':
      this.storage = config.storage;
      break;
    case 'string':
      this.storage = app.getResource('storages/' + config.storage);
      break;
    default:
      throw new Error("The 'session' middleware requires a storage to be passed in config.");
  }

  this.className = this.constructor.name;

  protos.util.onlySetEnumerable(this, ['className', 'storage']);
    
}

Session.prototype.storage = null;

/**
  Method that overrides OutgoingMessage::end, to automatically
  save sessions when changed.

  @private
 */

Session.prototype.endResponse = function() {
  var self = this;
  var req = this.request;
  var args = slice.call(arguments, 0);
  var end = this.constructor.prototype.end;
  if (req.sessionChanged()) {
    req.saveSessionState(function() {
      end.apply(self, args);
    });
  } else {
    end.apply(this, args);
  }
}

/**
  Creates a session

  @param {object} req
  @param {object} res
  @param {object} data
  @param {boolean} persistent
  @param {function} callback
  @public
*/

Session.prototype.create = function(req, res, data, persistent, callback) {
  var expires, guest, hashes, multi, self = this;
  guest = null;
  if (persistent == 'guest') {
    guest = true;
    persistent = true;
  }
  app.debug( guest ? 'Creating guest session' : 'Creating session' );
  hashes = this.createHash(guest);

  if (guest) {
    // Guest sessions have their own expiration timeout
    expires = this.config.guestExpires;
  } else {
    // Is the session persistent ? If yes, timeout should be permanentExpires
    // Otherwise, timeout should be temporaryExpires (browser session)
    expires = (persistent ? this.config.permanentExpires : this.config.temporaryExpires);
    
    // Set default expires if expires is zero
    if (!expires) expires = this.config.defaultExpires;
    
    data = _.extend(data, {
      fpr: hashes.fingerprint,
      pers: (persistent ? 1 : 0)
    });
  }

  multi = this.storage.multi();
  if (!guest && req.session && req.session.guest && req.hasCookie(this.config.sessCookie)) {
    multi.delete(req.getCookie(this.config.sessCookie));
  }
  multi.setHash(hashes.sessId, data);
  
  multi.expire(hashes.sessId, expires);

  multi.exec(function(err, replies) {
    if (err) {
      app.serverError(res, err);
    } else {

      // Expires has been calculated a few lines back
      res.setCookie(self.config.sessCookie, hashes.sessId, {
        expires: persistent ? expires : null,
        httpOnly: true
      });

      if (guest) {
        data.guest = parseInt(data.guest, 10);
      } else {
        res.setCookie(self.config.hashCookie, hashes.fingerprint, {
          expires: self.config.regenInterval,
          httpOnly: true
        });
      }

      data = self.typecast(data);
      
      req.session = data;
      
      req.__origSessionState = _.extend({}, data);
      req.__jsonSession = JSON.stringify(data);
      
      app.emit('load_session', hashes.sessId, req.session);
      
      callback.call(self, req.session, hashes, expires);
    }
  });
}

/**
  Destroys a session

  @param {object} req
  @param {object} res
  @param {function} callback
  @public
*/

Session.prototype.destroy = function(req, res, callback) {
  var fingerprint, sessId, self = this;
  app.debug('Destroying session');
  if (req.hasCookie(this.config.sessCookie) && req.session) {
    sessId = req.getCookie(this.config.sessCookie);
    fingerprint = this.getFingerprint(sessId);
    
    if (fingerprint == req.session.fpr) {
      this.storage.delete(sessId, function(err) {
        if (err) {
          app.serverError(res, err);
        } else {
          res.removeCookies(self.config.sessCookie, self.config.hashCookie);
          callback.call(self);
        }
      });
    } else {
      res.removeCookies(this.config.sessCookie, this.config.hashCookie);
      app.login(res);
    }
  } else {
    app.login(res);
  }
}

/**
  Loads the user session and passes control to next route.
  
  This function will automatically load the session, and call req.next()
  after it has been loaded, to pass control to the next route callback.
  
  Example:
  
    get('/login', app.session.load, function(req, res) {
      res.render('login');
    });
    
  @param {object} req
  @param {object} res
 */
 
Session.prototype.load = function(req, res) {
  (this instanceof Session ? this : app.session).loadSession(req, res, req.next);
}

/**
  Loads the session

  @param {object} req
  @param {object} res
  @param {function} callback
  @private
*/

Session.prototype.loadSession = function(req, res, callback) {
  
  var fingerprint, sessHash, sessId, self;

  if (req.__loadedSession === true) {
    app.debug("Trying to load session when it's already loaded...");
    callback.call(this);
    return;
  } else {
    req.session = {};
    req.__loadedSession = true;
  }

  self = this;

  sessId = req.getCookie(this.config.sessCookie);
  sessHash = req.getCookie(this.config.hashCookie);
  fingerprint = self.getFingerprint(sessId);

  if (sessId) { // A cookie with sessId exists

    // Get the session data from storage
    
    this.storage.getHash(sessId, function(err, data) {
      
      var expires, guest, hashes, multi, newHash, 
          newSess;

      if (err) { 
        // If errors retrieving session data,
        // respond with HTTP/500 & log error
        return app.serverError(res, err);
      }

      // If it's not a user session, it's a guest session
      guest = (data && data.user == null);

      if (_.isEmpty(data)) { // If data is empty

        if (self.config.guestSessions) {
          // => Create guest session
          res.removeCookie(self.config.hashCookie);
          self.createGuestSession(req, res, callback);
        } else {
          // => Remove cookies, contain invalid or expired session data
          req.removeCookies(self.config.sessCookie, self.config.hashCookie);
          callback.call(self, {});
        }

      } else { // Data is not empty

        if (guest) data.guest = parseInt(data.guest, 10);
        else  data.pers = parseInt(data.pers, 10);

        data = self.typecast(data);

        if (guest) { // If guest session detected

          app.debug('Loading guest session');
          req.session = data;
          req.__origSessionState = _.extend({}, data);
          req.__jsonSession = JSON.stringify(data);
          app.emit('load_session', sessId, req.session);
          callback.call(self);

          } else if (sessHash) { // If session hash detected

            if (sessHash == fingerprint && sessHash == data.fpr) { // If cookie hash matches session hash

              // Load user session
              app.debug('Loading session');
              req.session = data;
              req.__origSessionState = _.extend({}, data);
              req.__jsonSession = JSON.stringify(data);
              app.emit('load_session', sessId, req.session);
              callback.call(self);

            } else { // Else if session hash doesn't match

              // Remove the cookies on client, and redirect to login page
              req.removeCookies(self.config.sessCookie, self.config.hashCookie);
              app.login(res);

            }

          } else { // Else (session hash not detected)

            if (req.__noSessionRegenerate) { // Don't regenerate session (if set)

              // Load user session
              app.debug('Loading session (no regeneration)');
              req.session = data;
              req.__origSessionState = _.extend({}, data);
              req.__jsonSession = JSON.stringify(data);
              app.emit('load_session', sessId, req.session);
              callback.call(self);
              
            } else { // Session should be regenerated
              
              hashes = self.createHash();
              newSess = hashes.sessId;
              newHash = hashes.fingerprint;

              /*
              Is data persistent ? => use permanentExpires
              Otherwise:
              -- Is it a user session ? => use temporaryExpires
              -- Otherwise => use guestExpires
              */

              expires = self.config[(data.pers ? 'permanentExpires' : (data.user ? 'temporaryExpires' : 'guestExpires'))];

              // Set defaultExpires if expires is zero
              if (!expires) expires = self.config.defaultExpires;

              multi = self.storage.multi();
              multi.updateHash(sessId, {fpr: newHash});
              multi.rename(sessId, newSess);
              multi.expire(newSess, expires);

              multi.exec(function(err, replies) {
                if (err) {
                  app.serverError(res, err);
                } else {
                  res.setCookie(self.config.sessCookie, newSess, {
                    expires: data.pers ? expires : null,
                    httpOnly: true
                  });
                  res.setCookie(self.config.hashCookie, newHash, {
                    expires: self.config.regenInterval,
                    httpOnly: true
                  });
                  req.cookies[self.config.sessCookie.toLowerCase()] = newSess;
                  data.fpr = req.cookies[self.config.hashCookie.toLowerCase()] = newHash;
                  req.session = data;
                  req.__origSessionState = _.extend({}, data);
                  req.__jsonSession = JSON.stringify(data);
                  app.emit('load_session', sessId, req.session);
                  app.debug('Regenerating session');
                  callback.call(self);
                }
              });

            }

          }
        }

      });

    } else if (this.config.guestSessions) { // If guestSessions are enabled

      res.removeCookie(this.config.hashCookie);
      this.createGuestSession(req, res, callback);

    } else { // sessId is not present

      // Remove session hash if it's present
      if (sessHash) res.removeCookie(this.config.hashCookie);

      req.session = req.__origSessionState = {};
      req.__jsonSession = '';
      
      app.emit('load_session', sessId, req.session);
      callback.call(self);
    }
  }

/**
  Creates a guest session

  @param {object} req
  @param {object} res
  @param {object} data (optional)
  @param {function} callback
  @public
*/

Session.prototype.createGuestSession = function(req, res, data, callback) {
  var self = this;
  
  if (!callback) { 
    callback = data; 
    data = {guest: '1'};
  } else {
    data.guest = '1';
  }
  
  this.create(req, res, data, 'guest', function(data, hashes, expires) {
    return callback.call(self, data, hashes, expires);
  });
}

/**
  Generates a session ID
  
  @private
*/

Session.prototype.generateSid = app.uuid;

/**
  Generates a session fingerprint for a given session ID

  @param {object} req
  @param {string} sessId
  @returns {string} fingerprint hash
  @private
*/

Session.prototype.getFingerprint = function(sessId) {
  return app.md5(sessId + this.config.salt);
}

/**
  Creates a session hash

  @param {boolean} guest
  @returns {object}
  @private
*/

Session.prototype.createHash = function(guest) {
  var sessId = this.generateSid();
  if (guest) {
    return {sessId: sessId};
  } else {
    return {
      sessId: sessId,
      fingerprint: this.getFingerprint(sessId)
    }
  }
}

/**
  Performs automatic type coercion on session data.

  The session variables that will be converted, are specified in the `typecastVars` array.

  @param {object} data
  @returns {object} with data converted
  @private
*/

Session.prototype.typecast = function(data) {
  var tvars = this.config.typecastVars;
  for (var key,i=0,len=tvars.length; i < len; i++) {
    key = tvars[i];
    if (data[key] != null) data[key] = protos.util.typecast(data[key]);
  }
  return data;
}

function time(str) {
  str = str.split(/\s+/);
  return moment.duration(parseInt(str[0], 10), str[1]).asSeconds();
}

module.exports = Session;