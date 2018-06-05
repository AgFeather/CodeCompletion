/*globals YAHOO */
XC.Test.Core = new YAHOO.tool.TestCase({
  name: 'XC.Core Tests',

  testDebug: function() {
    var Assert = YAHOO.util.Assert;
    Assert.isFunction(XC.debug, 'XC.debug does not exist');
  },

  testLog: function() {
    var Assert = YAHOO.util.Assert;
    Assert.isFunction(XC.log, 'XC.log does not exist');
  },

  testWarn: function() {
    var Assert = YAHOO.util.Assert;
    Assert.isFunction(XC.warn, 'XC.warn does not exist');
  },

  testGroup: function() {
    var Assert = YAHOO.util.Assert;
    Assert.isFunction(XC.group, 'XC.group does not exist');
  },

  testGroupEnd: function() {
    var Assert = YAHOO.util.Assert;
    Assert.isFunction(XC.groupEnd, 'XC.groupEnd does not exist');
  },

  testIsFunction: function () {
    var Assert = YAHOO.util.Assert;
    Assert.isFunction(XC.isFunction, 'XC.isFunction does not exist');
    Assert.isTrue(XC.isFunction(function () {}), 'function is not a function');
    Assert.isTrue(XC.isFunction(XC.isFunction), 'function should be a function');
    Assert.isFalse(XC.isFunction(XC.Base.extend(Function.prototype)), 'object extending from Function.prototype should not be a function- should be an object.');
    Assert.isFalse(XC.isFunction({}), 'object should not be a function');
    Assert.isFalse(XC.isFunction(null), 'null should not be a function');
    Assert.isFalse(XC.isFunction(undefined), 'undefined should not be a function');
  },

  testIsString: function () {
    var Assert = YAHOO.util.Assert;
    Assert.isFunction(XC.isString, 'XC.isString does not exist.');
    Assert.isTrue(XC.isString(new String()), 'String constructor isn\'t a string.');
    Assert.isFalse(XC.isString(XC.Base.extend(String.prototype)), 'object extending from String.prototype should not be a string- should be an object');
    Assert.isTrue(XC.isString(''), 'Empty string is not a string');
    Assert.isTrue(XC.isString('something'), '"something" is not a string');
    Assert.isFalse(XC.isString(null), 'null should not be a string');
    Assert.isFalse(XC.isString(undefined), 'undefined should not be a string');
  }

});

YAHOO.tool.TestRunner.add(XC.Test.Core);
