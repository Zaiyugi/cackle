/* --- DataObject class declaration ---
 * Spec: Class for managing both host and device pointers
 */

#ifndef __DATAOBJECTDEFS_H__
#define __DATAOBJECTDEFS_H__

namespace akasha
{

/* Datatype Policies */
template < typename T >
struct DataObjectType;

/* Ownership Policies */
template <typename T>
struct CreateUnique;

/* Class for storing both a host and device pointer */
template < typename T, typename PointerOwnershipPolicy = CreateUnique<T> >
class DataObject;

}

#endif
