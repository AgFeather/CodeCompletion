/*!
 * Основные константы
 */

/** @const */
Collection.prototype.NAME = 'Collection';

/** @const */
Collection.prototype.VERSION = [5, 5, 5];

const NULL = {};
const CACHE_VERSION = 14;
const CLUSTER = [
	'namespace',
	'collection',
	'filter',
	'context',
	'var'
];

/**
 * Вернуть строку: название + версия библиотеки
 * @return {string}
 */
Collection.prototype.collection = function () {
	return this.NAME + ' ' + this.VERSION.join('.');
};

//#include ./const.hack.js
