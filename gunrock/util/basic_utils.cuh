// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * basic_utils.cuh
 *
 * @brief Basic Utilities for the kernel
 */


namespace gunrock {
namespace util {


	// Struct to create an unsigned version of an integral type 
	template<typename T> struct make_unsigned;
	
	template<> struct make_unsigned<char>                   { typedef unsigned char          type; };
	template<> struct make_unsigned<signed char>            { typedef signed   char          type; };
	template<> struct make_unsigned<unsigned char>          { typedef unsigned char          type; };
	template<> struct make_unsigned<short>                  { typedef unsigned short         type; };
	template<> struct make_unsigned<unsigned short>         { typedef unsigned short         type; };
	template<> struct make_unsigned<int>                    { typedef unsigned int           type; };
	template<> struct make_unsigned<unsigned int>           { typedef unsigned int           type; };
	template<> struct make_unsigned<long int>               { typedef unsigned long int      type; };
	template<> struct make_unsigned<unsigned long int>      { typedef unsigned long int      type; };
	template<> struct make_unsigned<long long int>          { typedef unsigned long long int type; };
	template<> struct make_unsigned<unsigned long long int> { typedef unsigned long long int type; };

	
	// Struct to create a typename that is compatible with CUDA atomic functions.
	// Most atomic functions take only int, unsigned int and unsigned long long int 
        template<typename T> struct get_atomic_type;

        template<> struct get_atomic_type<int>                    { typedef int 		   type; };
        template<> struct get_atomic_type<unsigned int>           { typedef unsigned int           type; };
        template<> struct get_atomic_type<long int>               { typedef unsigned long int      type; };
        template<> struct get_atomic_type<unsigned long int>      { typedef unsigned long int      type; };
        template<> struct get_atomic_type<long long int>          { typedef unsigned long long int type; };
        template<> struct get_atomic_type<unsigned long long int> { typedef unsigned long long int type; };


}
}
