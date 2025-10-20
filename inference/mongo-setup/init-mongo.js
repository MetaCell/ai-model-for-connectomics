db = db.getSiblingDB('synister_fafb_v3'); // replace with your app DB name

db.createUser({
  user: 'admin',
  pwd: 'example',
  roles: [
    {
      role: 'readWrite',
      db: 'synister_fafb_v3'
    }
  ]
});
