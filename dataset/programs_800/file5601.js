/**
 * Configuration loader module
 *
 * This merges the base config with the environment config and exposes them.
 */
'use strict';
var _  = require('lodash');

var rawConfig = require('./../config/config.js');
var packageJson = require('../package.json');

// Generate the env var rather than getting it from express
//
var env = process.env.NODE_ENV || 'development';

// Define defaultsDeep method (from http://lodash.com/docs#partialRight)
// http://expressjs.com/api.html#app-settings
var defaultsDeep = _.partialRight(_.merge, function deep(value, other) {
    return _.merge(value, other, deep);
});

// Load the config and set the defaults
module.exports = defaultsDeep(rawConfig[env], rawConfig.default);

// Set additional settings
module.exports.environment = env;
module.exports.version = packageJson.version;
